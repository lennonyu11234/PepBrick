import os.path
import shutil

import torch
from torch import nn
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.loader import DataLoader as DataLoaderGraph
from Module.Cyclic_utils import add_explicit_hydrogens
from torch_geometric.data import Data
import numpy as np

from args import args
from GNN_Scoring.GNN_module import GNNEncoder
from rdkit import Chem
from rdkit.Chem import AllChem
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT]}


def mol_to_graph_data_with_batch(mol):
    atom_feature_list = []
    atom_idx_map = {}
    reduced_atom_idx = 0
    non_hydrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                           [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
            atom_feature_list.append(atom_feature)
            atom_idx_map[atom.GetIdx()] = reduced_atom_idx
            reduced_atom_idx += 1
    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)

    num_bond_features = 2
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edges_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i in non_hydrogen_atoms and j in non_hydrogen_atoms:
                if mol.GetAtomWithIdx(i).GetAtomicNum() != 1 and mol.GetAtomWithIdx(j).GetAtomicNum() != 1:
                    i_reduced = atom_idx_map[i]
                    j_reduced = atom_idx_map[j]

                    edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                                   [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
                    edges_list.append((i_reduced, j_reduced))
                    edges_features_list.append(edge_feature)
                    edges_list.append((j_reduced, i_reduced))
                    edges_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edges_features_list), dtype=torch.long)

        batch = torch.zeros(len(x))

    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    return graph_data



def check_valid_smiles(smiles_list):
    scores = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and AllChem.EmbedMolecule(mol) is not None:
                scores.append(0.3)
            else:
                scores.append(-0.3)
        except Exception:
            scores.append(-0.3)
    return torch.tensor(scores)


def normalize_to_range(tensor, min_val, max_val):
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    normalized_tensor = normalized_tensor - 0.5
    normalized_tensor = torch.clamp(normalized_tensor, -0.5, 0.5)

    return -normalized_tensor


class ScoringFunction(nn.Module):
    def __init__(self, name, func):
        super().__init__()
        self.name = name
        self.func = func
        self.gnn_reg = GNNEncoder(num_layer=args.gnn_num_layer,
                                  emb_dim=args.gnn_emb_dim,
                                  JK=args.JK,
                                  drop_ratio=args.dropout,
                                  graph_pooling=args.graph_pooling,
                                  gnn_type=args.gnn_type,
                                  func='regression').to(device)

        self.gnn_cls = GNNEncoder(num_layer=args.gnn_num_layer,
                                  emb_dim=args.gnn_emb_dim,
                                  JK=args.JK,
                                  drop_ratio=args.dropout,
                                  graph_pooling=args.graph_pooling,
                                  gnn_type=args.gnn_type,
                                  func='classification').to(device)

        if self.name == 'Cyclic_reg':
            self.gnn_reg.load_state_dict(torch.load('Data\Model\cyc\GIN_reg.pth'))
        elif self.name == 'Opioid':
            self.gnn_reg.load_state_dict(torch.load(r'Data\Model\opioid\GIN_reg.pth'))
        elif self.name == 'Neuro':
            self.gnn_cls.load_state_dict(torch.load(r'Data\Model\Neuro/GIN_cls.pth'))
        elif self.name == 'linear':
            self.gnn_cls.load_state_dict(torch.load(r'Data\Model\lin/GIN_cls.pth'))
        elif self.name == '7W41':
            self.gnn_reg.load_state_dict(torch.load(r'Data\Model/7W14_demo112/GraphSAGE_reg.pth'))

    def forward(self, graph_data):
        if self.func == 'regression':
            self.gnn_reg.eval()
            x, edge_index, edge_attr, batch = graph_data.x.to(device), \
                graph_data.edge_index.to(device), \
                graph_data.edge_attr.to(device), \
                graph_data.batch.to(device)
            _, _, _, logit = self.gnn_reg.forward(x, edge_index, edge_attr, batch)
            prediction = logit.squeeze()
            if self.name in ('Opioid', '7W41'):
                min_value = -12
                max_value = -4
                normalized_output = -(2 * ((prediction - min_value) / (max_value - min_value)) - 1)
                return prediction, normalized_output
            elif self.name == 'Cyclic_reg':
                min_value = -10
                max_value = -4
                normalized_output = (2 * ((prediction - min_value) / (max_value - min_value)) - 1)
                return prediction, normalized_output

        elif self.func == 'classification':
            self.gnn_cls.eval()
            x, edge_index, edge_attr, batch = graph_data.x.to(device), \
                graph_data.edge_index.to(device), \
                graph_data.edge_attr.to(device), \
                graph_data.batch.to(device)
            _, _, _, logit = self.gnn_cls.forward(x, edge_index, edge_attr, batch)
            prediction = torch.argmax(logit, dim=1)
            scores = torch.where(prediction == 1, torch.tensor(0.3), torch.tensor(-0.3))

            return prediction, scores


def calculate_model_score(smiles_list,
                          scoring_model):
    scores, true_scores, valid_mask = [], [], []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                try:
                    smiles_h = add_explicit_hydrogens(smiles)
                    mol = Chem.MolFromSmiles(smiles_h)
                except Exception:
                    mol = None

            if mol is None:
                raise ValueError("Invalid molecule")
            graph_data = mol_to_graph_data_with_batch(mol)
            with torch.no_grad():
                scoring_model.eval()
                true_score, output = scoring_model(graph_data)
            if scoring_model.func == 'regression':
                scores.append(output.item())
                true_scores.append(true_score)
            else:
                scores.append(output.item())
                true_scores.append(true_score)
            valid_mask.append(True)
        except Exception as e:
            scores.append(-0.2)
            true_scores.append(0.0)
            valid_mask.append(False)

    return (torch.tensor(scores, device=device),
            torch.tensor(true_scores, device=device),
            torch.tensor(valid_mask, device=device))







