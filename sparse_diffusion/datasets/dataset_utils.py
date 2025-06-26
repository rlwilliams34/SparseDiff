import os.path as osp
import pickle
from typing import Any, Sequence

from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import networkx as nx



def graph_to_pyg_data(g: nx.Graph, bins=None, n_bins = 10, global_max=None, global_min=None):
    # --- NODE FEATURES ---
    num_nodes = g.number_of_nodes()
    x = torch.ones((num_nodes, 1))  # "null" node feature
    
    # --- EDGE INDEX ---
    edge_list = list(g.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
    
    # --- EDGE ATTRIBUTES ---
    edge_weights = np.array([g[u][v]['weight'] for u, v in edge_list])
    
    # Normalize edge weights globally or locally
    if global_min is None:
        global_min = edge_weights.min()
    if global_max is None:
        global_max = edge_weights.max()
        
    #norm_weights = (edge_weights - global_min) / (global_max - global_min + 1e-8)
    
    # Discretize into bins
    if bins is None:
        bins = np.linspace(global_min, global_max, n_bins + 1)
    
    digitized = np.digitize(edge_weights, bins) - 1  # convert to 0-based bin index
    digitized = np.clip(digitized, 0, n_bins - 1)
    
    # One-hot encode edge features
    edge_attr = torch.eye(n_bins)[digitized]
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    node_types = []
    all_charge = []
    for atom in mol.GetAtoms():
        node_types.append(atom_encoder[atom.GetSymbol()])
        all_charge.append(atom.GetFormalCharge())

    node_types = torch.Tensor(node_types).long()
    all_charge = torch.Tensor(all_charge).long()

    data = Data(
        x=node_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        charge=all_charge,
        smiles=smiles,
    )
    return data


def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(
        to_keep,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=len(to_keep),
    )
    return Data(
        x=data.x[to_keep] - 1,  # Shift onehot encoding to match atom decoder
        charge=data.charge[to_keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def files_exist(files) -> bool:
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class Statistics:
    def __init__(
        self, num_nodes, node_types, bond_types, charge_types=None, valencies=None
    ):
        self.num_nodes = num_nodes
        self.node_types = node_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data
