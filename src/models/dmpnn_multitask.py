"""Small bond-directed message passing network for SI and concentration tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F


@dataclass
class MoleculeGraph:
    atoms: np.ndarray
    bonds: np.ndarray
    source: np.ndarray
    target: np.ndarray
    reverse: np.ndarray


def _one_hot(value, choices):
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def featurize_smiles(smiles: str) -> MoleculeGraph:
    """Build fixed categorical atom/bond features and directed edges."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid SMILES: {smiles}")
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            _one_hot(atom.GetAtomicNum(), [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])
            + _one_hot(atom.GetDegree(), list(range(6)))
            + _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + [float(atom.GetIsAromatic())]
            + _one_hot(atom.GetTotalNumHs(), list(range(5)))
        )
    bonds, sources, targets, reverses = [], [], [], []
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                  Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        features = (_one_hot(bond.GetBondType(), bond_types)
                    + [float(bond.GetIsConjugated()), float(bond.IsInRing())])
        index = len(sources)
        sources.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        targets.extend([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        reverses.extend([index + 1, index])
        bonds.extend([features, features])
    return MoleculeGraph(
        np.asarray(atom_features, dtype=np.float32),
        np.asarray(bonds, dtype=np.float32).reshape(-1, 7),
        np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64),
        np.asarray(reverses, dtype=np.int64),
    )


def batch_graphs(graphs: list[MoleculeGraph], device=None):
    atoms, bonds, source, target, reverse, atom_batch = [], [], [], [], [], []
    atom_offset = edge_offset = 0
    for i, graph in enumerate(graphs):
        atoms.append(graph.atoms)
        atom_batch.extend([i] * len(graph.atoms))
        if len(graph.source):
            bonds.append(graph.bonds)
            source.append(graph.source + atom_offset)
            target.append(graph.target + atom_offset)
            reverse.append(graph.reverse + edge_offset)
            edge_offset += len(graph.source)
        atom_offset += len(graph.atoms)
    tensors = [
        torch.as_tensor(np.concatenate(atoms), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(bonds) if bonds else np.zeros((0, 7), np.float32), dtype=torch.float32, device=device),
        torch.as_tensor(np.concatenate(source) if source else np.empty(0, np.int64), dtype=torch.long, device=device),
        torch.as_tensor(np.concatenate(target) if target else np.empty(0, np.int64), dtype=torch.long, device=device),
        torch.as_tensor(np.concatenate(reverse) if reverse else np.empty(0, np.int64), dtype=torch.long, device=device),
        torch.as_tensor(atom_batch, dtype=torch.long, device=device),
    ]
    return tensors


class DMPNNMultiTask(nn.Module):
    """Shared directed-message encoder with raw SI and two log-concentration heads."""

    atom_features = 32
    bond_features = 7

    def __init__(self, hidden_size: int = 64, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        self.input = nn.Linear(self.atom_features + self.bond_features, hidden_size)
        self.message = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = nn.Linear(self.atom_features + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, graph_tensors):
        atoms, bonds, source, target, reverse, atom_batch = graph_tensors
        n_graphs = int(atom_batch.max().item()) + 1
        if len(source):
            initial = F.relu(self.input(torch.cat([atoms[source], bonds], dim=1)))
            hidden = initial
            for _ in range(self.depth - 1):
                incoming = hidden.new_zeros((len(atoms), hidden.shape[1]))
                incoming.index_add_(0, target, hidden)
                message = incoming[source] - hidden[reverse]
                hidden = F.relu(initial + self.message(message))
            incoming = hidden.new_zeros((len(atoms), hidden.shape[1]))
            incoming.index_add_(0, target, hidden)
        else:
            incoming = atoms.new_zeros((len(atoms), self.message.out_features))
        atom_state = F.relu(self.readout(torch.cat([atoms, incoming], dim=1)))
        pooled = atom_state.new_zeros((n_graphs, atom_state.shape[1]))
        pooled.index_add_(0, atom_batch, atom_state)
        counts = torch.bincount(atom_batch, minlength=n_graphs).clamp_min(1).to(pooled.dtype)
        output = self.output(self.dropout(pooled / counts[:, None]))
        # SI cannot be negative; concentration heads remain unconstrained in log space.
        return torch.cat([F.softplus(output[:, :1]), output[:, 1:]], dim=1)


def bounded_mse(prediction, target, qualifier):
    """Squared error for exact logs; one-sided loss when the observation is censored."""
    qualifier = np.asarray(qualifier, dtype=object)
    pred = prediction
    target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    loss = (pred - target).square()
    lower = torch.as_tensor([str(q).strip().startswith((">", ">=")) for q in qualifier],
                            device=pred.device)
    upper = torch.as_tensor([str(q).strip().startswith(("<", "<=")) for q in qualifier],
                            device=pred.device)
    loss = torch.where(lower, F.relu(target - pred).square(), loss)
    loss = torch.where(upper, F.relu(pred - target).square(), loss)
    return loss
