import os
import pathlib
import os.path as osp

import numpy as np
from tqdm import tqdm
import networkx as nx
import torch
import pickle as pkl
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd
from networkx import to_numpy_array
from torch_geometric.loader import DataLoader
from sparse_diffusion.utils import PlaceHolder
from sparse_diffusion.datasets.abstract_dataset import (
    AbstractDataModule,
    AbstractDatasetInfos,
)
from sparse_diffusion.datasets.dataset_utils import (
    load_pickle,
    save_pickle,
    Statistics,
    to_list,
    RemoveYTransform,
)
from sparse_diffusion.metrics.metrics_utils import (
    node_counts,
    edge_counts,
)

import pytorch_lightning as pl
from sparse_diffusion.utils import PlaceHolder
import os, pickle, torch
from torch_geometric.data import InMemoryDataset, Data
from sparse_diffusion.datasets.dataset_utils import graph_to_pyg_data  # If you move your conversion here
from glob import glob
from sparse_diffusion.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from sparse_diffusion.diffusion.distributions import DistributionNodes

#from sparse_diffusion.datasets.extra_features import DummyExtraFeatures
#from sparse_diffusion.metrics.abstract_metrics import TrainAbstractMetricsDiscrete



class LobsterDataset(InMemoryDataset):
    def __init__(self, root, split='train', n_bins=10, transform=None, pre_transform=None):
        self.split = split
        self.n_bins = n_bins
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.midpoints = None
        midpoint_path = os.path.join(self.processed_dir, f'{self.split}_midpoints.npy')
        if os.path.exists(midpoint_path):
            self.midpoints = np.load(midpoint_path)
        

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def process(self):
        raw_path = os.path.join(self.root, f'{self.split}-graphs.pkl')
        with open(raw_path, 'rb') as f:
            nx_graphs = pickle.load(f)

        # Optionally compute global min/max across all weights
        all_weights = [g[u][v]['weight'] for g in nx_graphs for u, v in g.edges()]
        gmin, gmax = min(all_weights), max(all_weights)
        bins = np.linspace(gmin, gmax, self.n_bins + 1)
        midpoints = 0.5 * (bins[:-1] + bins[1:])
        self.midpoints = midpoints
        np.save(os.path.join(self.processed_dir, f'{self.split}_midpoints.npy'), self.midpoints)

        pyg_graphs = [graph_to_pyg_data(g, bins = bins, n_bins=self.n_bins, global_min=gmin, global_max=gmax) for g in nx_graphs]

        data, slices = self.collate(pyg_graphs)
        torch.save((data, slices), self.processed_paths[0])
    
    def get_midpoints(self):
        return self.midpoints



class LobsterDataModule(AbstractDataModule):
    def __init__(self, cfg):
        from .lobster import LobsterDataset
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.n_bins = cfg.dataset.n_bins
        datasets = {
            "train": LobsterDataset(cfg.dataset.root, split="train", n_bins=self.n_bins),
            "val": LobsterDataset(cfg.dataset.root, split="val", n_bins=self.n_bins),
            "test": LobsterDataset(cfg.dataset.root, split="test", n_bins=self.n_bins),
        }
        super().__init__(cfg, datasets)
        self.dataset_stat()
    
    def dataset_stat(self):
        self.statistics = {"train": {}, "val": {}, "test": {}}
        for split_name, dataset in [("train", self.train_dataset),
                                    ("val", self.val_dataset),
                                    ("test", self.test_dataset)]:
            num_nodes = {}
            for graph in dataset:
                n = graph.num_nodes
                num_nodes[n] = num_nodes.get(n, 0) + 1
            self.statistics[split_name]["num_nodes"] = num_nodes

class LobsterInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.name = "lobster"
        self.is_molecular = False
        self.use_charge = False
        self.remove_h = False # ???
        self.num_edge_types = datamodule.n_bins
        
        
        self.num_node_types = 1
        self.num_edge_types = datamodule.n_bins
        self.num_charge_types = 0
        self.node_types = torch.tensor([1.0])
        self.edge_types = torch.tensor([1.0] * self.num_edge_types)
        self.charge_types = torch.tensor([])
        
        
        self.output_dims = PlaceHolder(X=self.num_node_types, charge=self.num_charge_types, E=self.num_edge_types, y=0)
        train_n_nodes = datamodule.statistics["train"]["num_nodes"]
        val_n_nodes = datamodule.statistics["val"]["num_nodes"]
        test_n_nodes = datamodule.statistics["test"]["num_nodes"]
        max_n_nodes = max(
            max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys())
        )
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value
        self.n_nodes = n_nodes / n_nodes.sum()

        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

        
    
    
    
















