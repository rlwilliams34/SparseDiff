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
from sparse_diffusion.datasets.utils import graph_to_pyg_data  # If you move your conversion here
from glob import glob

def tree_top_generator(l, py_random):
    """
    Generates a random bifurcating tree with l leaves.
    More efficient than scanning the whole node list each time.
    """
    g = nx.Graph()

    # Start with a root node connected to two leaves
    g.add_edges_from([(0, 1), (0, 2)])
    next_node = 3
    active_leaves = [1, 2]

    for _ in range(l - 2):
        # Pick a random active leaf
        selected = py_random.choice(active_leaves)
        active_leaves.remove(selected)

        # Add two new leaves
        left = next_node
        right = next_node + 1
        next_node += 2

        g.add_edges_from([(selected, left), (selected, right)])
        active_leaves.extend([left, right])

    return g


def tree_generator(num_leaves, seed = 285):
    '''
    Generates requested number of bifurcating trees
    Args:
        n: number of leaves
        num_graphs: number of requested graphs
    '''
    num_leaves = 750
    npr = np.random.RandomState(seed)
    py_random = random.Random(seed)
    g = tree_top_generator(num_leaves, py_random)
    mu = npr.uniform(7, 13)
    weights = npr.gamma(mu * mu, 1 / mu, g.number_of_edges())

    # Add weights directly to edges
    for ((n1, n2), w) in zip(g.edges(), weights):
        g[n1][n2]['weight'] = w
    return g


class LobsterDataset(InMemoryDataset):
    def __init__(self, root, split='train', n_bins=100, num_leaves=1000, seed=285, transform=None, pre_transform=None):
        self.split = split
        self.n_bins = n_bins
        self.num_leaves = num_leaves
        self.seed = seed
        self.midpoints = None
        super().__init__(root, transform, pre_transform)

        # Load saved data if it exists
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Load midpoints if they were saved
        midpoint_path = os.path.join(self.processed_dir, f'{self.split}_midpoints.npy')
        if os.path.exists(midpoint_path):
            self.midpoints = np.load(midpoint_path)
        else:
            print(f"[Warning] Midpoints not found for {self.split} — did you call process()?")

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def process(self):
        # Generate a synthetic tree with edge weights
        G = tree_generator(num_leaves=self.num_leaves, seed=self.seed)
        print(f"[{self.split}] Generated tree with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Get all edge weights and compute bins
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        gmin, gmax = min(weights), max(weights)
        bins = np.linspace(gmin, gmax, self.n_bins + 1)
        self.midpoints = 0.5 * (bins[:-1] + bins[1:])
        np.save(os.path.join(self.processed_dir, f'{self.split}_midpoints.npy'), self.midpoints)

        # Convert NetworkX graph to PyG Data object
        pyg_graph = graph_to_pyg_data(
            G,
            bins=bins,
            n_bins=self.n_bins,
            global_min=gmin,
            global_max=gmax,
        )

        # Collate and save
        data, slices = self.collate([pyg_graph])
        torch.save((data, slices), self.processed_paths[0])

    def get_midpoints(self):
        return self.midpoints





















