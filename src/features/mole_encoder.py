"""MolE GIN encoder used to extract the pretrained graph representation.

The graph and encoder contract follows rolayoalarcon/MolE, commit vendored in
the project documentation: gin_concat, five 200-dimension layers, sum pooling.
Only the first value returned by the upstream forward pass is needed; the 8k
projection head is intentionally omitted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops


ATOM_COUNT = 119
CHIRALITY = (
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
)
BONDS = (Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC)
BOND_DIRECTIONS = (Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
                   Chem.rdchem.BondDir.ENDDOWNRIGHT)


class GINEConv(MessagePassing):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim), nn.ReLU(),
        )
        self.edge_embedding1 = nn.Embedding(5, emb_dim)
        self.edge_embedding2 = nn.Embedding(3, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        self_loops = torch.zeros((x.size(0), 2), dtype=edge_attr.dtype, device=edge_attr.device)
        self_loops[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loops), dim=0)
        edge = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class MolEEncoder(nn.Module):
    """Pretrained MolE graph trunk; output matches upstream ``h_global_embedding``."""

    def __init__(self, num_layer: int = 5, emb_dim: int = 200, drop_ratio: float = 0.0):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.x_embedding1 = nn.Embedding(ATOM_COUNT, emb_dim)
        self.x_embedding2 = nn.Embedding(3, emb_dim)
        self.gnns = nn.ModuleList(GINEConv(emb_dim) for _ in range(num_layer))
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(emb_dim) for _ in range(num_layer))

    @property
    def output_dim(self) -> int:
        return self.num_layer * self.emb_dim

    def forward(self, data):
        h_init = self.x_embedding1(data.x[:, 0]) + self.x_embedding2(data.x[:, 1])
        layers = []
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_init if layer == 0 else layers[-1], data.edge_index, data.edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(h if layer == self.num_layer - 1 else F.relu(h),
                          self.drop_ratio, training=self.training)
            layers.append(h)
        return torch.cat([global_add_pool(h, data.batch) for h in layers], dim=1)


def mol_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES for MolE: {smiles!r}")
    mol = Chem.AddHs(mol)
    atom_types, chiralities = [], []
    for atom in mol.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        try:
            chirality = CHIRALITY.index(atom.GetChiralTag())
        except ValueError as exc:
            raise ValueError(f"Unsupported MolE chirality in {smiles!r}") from exc
        if chirality >= 3:
            raise ValueError(f"MolE checkpoint cannot encode CHI_OTHER in {smiles!r}")
        chiralities.append(chirality)
    if max(atom_types, default=1) > ATOM_COUNT - 1:
        raise ValueError(f"Unsupported atomic number in {smiles!r}")
    x = torch.tensor(np.column_stack((np.asarray(atom_types) - 1, chiralities)), dtype=torch.long)
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            bond_type = BONDS.index(bond.GetBondType())
            direction = BOND_DIRECTIONS.index(bond.GetBondDir())
        except ValueError as exc:
            raise ValueError(f"Unsupported MolE bond feature in {smiles!r}") from exc
        row.extend((start, end))
        col.extend((end, start))
        edge_attr.extend(((bond_type, direction), (bond_type, direction)))
    return Data(x=x, edge_index=torch.tensor((row, col), dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.long).reshape(-1, 2))


def mol_batch(smiles: list[str], device: str):
    return Batch.from_data_list([mol_to_graph(value) for value in smiles]).to(device)
