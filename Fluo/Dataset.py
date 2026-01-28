import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs
from torch_geometric.data import Data, InMemoryDataset


@dataclass
class Args:
    seed: int = 42
    epochs: int = 500
    batch_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 1e-5
    emb_dim: int = 512
    num_layer: int = 3
    drop_ratio: float = 0.1
    JK: str = "last"
    pooling: str = "mean"

    solute_gnn: str = "GAT"
    solvent_gnn: str = "GAT"

    use_scaffold: bool = False
    scaffold_vocab: int = 2048
    scaffold_emb: int = 512

    fusion_mode: str = "concat"
    fusion_dim: int = 512

    morgan_bits: int = 2048
    morgan_radius: int = 2


args = Args()


def safe_index(lst, item, default=0):
    try:
        return lst.index(item)
    except ValueError:
        return default


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


num_atom_type = 120
num_chirality_tag = 4
num_bond_type = 7
num_bond_direction = 4

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def mol_to_morgan_fp(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2) -> torch.Tensor:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)


def mol_to_graph_data(mol: Chem.Mol) -> Data:
    atom_features = []
    atom_idx_map = {}
    idx = 0

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        atom_features.append([
            safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
            safe_index(allowable_features["possible_chirality_list"], atom.GetChiralTag())
        ])
        atom_idx_map[atom.GetIdx()] = idx
        idx += 1

    x = torch.tensor(atom_features, dtype=torch.long)

    edges = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in atom_idx_map or j not in atom_idx_map:
            continue

        i_r, j_r = atom_idx_map[i], atom_idx_map[j]
        e = [
            safe_index(allowable_features["possible_bonds"], bond.GetBondType()),
            safe_index(allowable_features["possible_bond_dirs"], bond.GetBondDir())
        ]

        edges.append((i_r, j_r))
        edges.append((j_r, i_r))
        edge_attrs.append(e)
        edge_attrs.append(e)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        assert edge_index.max() < x.size(0)
        assert edge_index.min() >= 0

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def murcko_scaffold_id(smiles: str, vocab_size: int) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if scaff is None:
        return 0
    smi = Chem.MolToSmiles(scaff, isomericSmiles=False)
    if smi == "":
        return 0

    import hashlib
    h = hashlib.md5(smi.encode("utf-8")).hexdigest()
    return int(h, 16) % vocab_size


class SoluteSolventData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "s_edge_index":
            return int(self.s_x.size(0))
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "s_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class SoluteSolventDataset(InMemoryDataset):
    def __init__(self, root: str, csv_path: str, transform=None, pre_transform=None):
        self.csv_path = csv_path
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["processed.dataset"]

    def process(self):
        df = pd.read_csv(self.csv_path, usecols=["smiles", "solvent", "plqy"])
        df = df.dropna()

        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            solute_smiles = str(row["smiles"])
            solvent_smiles = str(row["solvent"])
            label = float(row["plqy"])
            solute_mol = Chem.MolFromSmiles(solute_smiles)
            solvent_mol = Chem.MolFromSmiles(solvent_smiles)
            if solute_mol is None or solvent_mol is None:
                continue

            try:
                g0 = mol_to_graph_data(solute_mol)
                s0 = mol_to_graph_data(solvent_mol)
            except AssertionError:
                continue

            if g0.x.size(0) == 0 or s0.x.size(0) == 0:
                continue
            try:
                g_morgan = mol_to_morgan_fp(solute_mol, n_bits=args.morgan_bits, radius=args.morgan_radius).unsqueeze(
                    0)
                s_morgan = mol_to_morgan_fp(solvent_mol, n_bits=args.morgan_bits, radius=args.morgan_radius).unsqueeze(
                    0)

            except Exception:
                continue

            data = SoluteSolventData(
                x=g0.x,
                edge_index=g0.edge_index,
                edge_attr=g0.edge_attr,

                s_x=s0.x,
                s_edge_index=s0.edge_index,
                s_edge_attr=s0.edge_attr,

                g_morgan=g_morgan,
                s_morgan=s_morgan,

                scaffold_id=torch.tensor(
                    murcko_scaffold_id(solute_smiles, args.scaffold_vocab),
                    dtype=torch.long
                ),
                label=torch.tensor(label, dtype=torch.float32)
            )

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
