import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import numpy as np
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


def read_data(path):
    columns_to_read = ['SMILES_cyc']
    pep = pd.read_csv(filepath_or_buffer=path, usecols=columns_to_read, encoding='utf-8')
    pred_pep = [(row['SMILES_cyc']) for index, row in pep.iterrows()]
    return pred_pep


def read_data_labeled(path):
    columns_to_read = ['SMILES_cyc', 'affinity']
    pep = pd.read_csv(filepath_or_buffer=path, usecols=columns_to_read)
    pred_pep = [(row['SMILES_cyc'], row['affinity']) for index, row in pep.iterrows()]
    return pred_pep


def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)


def mol_to_graph_data(mol):
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

    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph_data


def mol_to_graph_data_with_batch(mol):
    atom_feature_list = []
    atom_idx_map = {}
    reduced_atom_idx = 0
    non_hydrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:  # Exclude hydrogen atoms
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


path = r'Data\Dataset'
labeled_data = read_data_labeled(path)


class DatasetWithLabel(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None):
        super(InMemoryDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_data.dataset']

    def download(self):
        pass

    def process(self):
        graph_data_list = []
        for idx, (smile, label) in enumerate(tqdm(labeled_data, desc='Processing')):
            try:
                molecule = Chem.MolFromSmiles(smile)
            except:
                continue

            data = mol_to_graph_data(molecule)

            data.label = torch.tensor(label, dtype=torch.float)
            graph_data_list.append(data)
        data, slice = self.collate(graph_data_list)
        torch.save((data, slice), self.processed_paths[0])






















