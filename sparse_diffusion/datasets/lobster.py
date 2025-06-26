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


import os, pickle, torch
from torch_geometric.data import InMemoryDataset, Data
from sparse_diffusion.datasets.dataset_utils import graph_to_pyg_data  # If you move your conversion here
from glob import glob



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





















