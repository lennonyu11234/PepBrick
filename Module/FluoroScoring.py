import torch
import numpy as np
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from torch_geometric.data import Data
import traceback

from fluorophore.FLSF.Fluore_Module import GraphSolventScaffoldRegressor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def safe_index(lst, item, default=0):
    try:
        return lst.index(item)
    except ValueError:
        return default

def mol_to_morgan_fp(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2) -> torch.Tensor:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)  # [n_bits]

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

    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in atom_idx_map or j not in atom_idx_map:
            continue

        i_r, j_r = atom_idx_map[i], atom_idx_map[j]
        e = [
            safe_index(allowable_features["possible_bonds"], bond.GetBondType()),
            safe_index(allowable_features["possible_bond_dirs"], bond.GetBondDir())
        ]
        edges.append((i_r, j_r)); edge_attrs.append(e)
        edges.append((j_r, i_r)); edge_attrs.append(e)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class SoluteSolventData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "s_edge_index":
            return int(self.s_x.size(0))
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "s_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

_WATER_CACHE = {}

def get_cached_water_graph_and_fp(morgan_bits=2048, morgan_radius=2):
    key = (morgan_bits, morgan_radius)
    if key in _WATER_CACHE:
        return _WATER_CACHE[key]

    water = Chem.MolFromSmiles("CS(C)=O")
    if water is None:
        raise ValueError('Failed to build water mol from SMILES="O"')

    s_graph = mol_to_graph_data(water)
    s_fp = mol_to_morgan_fp(water, n_bits=morgan_bits, radius=morgan_radius)  # [2048]

    _WATER_CACHE[key] = (s_graph, s_fp)
    return s_graph, s_fp


def build_data_with_fixed_water(solute_smiles: str,
                                morgan_bits=2048,
                                morgan_radius=2) -> SoluteSolventData:
    mol = Chem.MolFromSmiles(solute_smiles)
    if mol is None:
        raise ValueError(f"Invalid solute SMILES: {solute_smiles}")

    g_graph = mol_to_graph_data(mol)
    if g_graph.x.size(0) == 0:
        raise ValueError("Solute graph has 0 nodes (after removing H).")

    g_fp = mol_to_morgan_fp(mol, n_bits=morgan_bits, radius=morgan_radius)

    s_graph, s_fp = get_cached_water_graph_and_fp(morgan_bits=morgan_bits, morgan_radius=morgan_radius)

    data = SoluteSolventData(
        x=g_graph.x,
        edge_index=g_graph.edge_index,
        edge_attr=g_graph.edge_attr,
        batch=torch.zeros(g_graph.x.size(0), dtype=torch.long),

        s_x=s_graph.x,
        s_edge_index=s_graph.edge_index,
        s_edge_attr=s_graph.edge_attr,
        s_x_batch=torch.zeros(s_graph.x.size(0), dtype=torch.long),

        g_morgan=g_fp.unsqueeze(0),
        s_morgan=s_fp.unsqueeze(0),

        scaffold_id=torch.tensor(0, dtype=torch.long),
    )
    return data


class ScoringFunction(nn.Module):
    def __init__(self,
                 name,
                 y_min,
                 y_max):
        super().__init__()
        self.y_min = y_min
        self.y_max = y_max

        self.model = GraphSolventScaffoldRegressor(emb_dim=512,
                                                   num_layer=3,
                                                   drop_ratio=0.1,
                                                   JK="last",
                                                   pooling="mean",
                                                   solute_gnn="GIN",
                                                   solvent_gnn="GIN",
                                                   use_scaffold=False,
                                                   scaffold_vocab=2048,
                                                   scaffold_emb=512,
                                                   fusion_mode="concat",
                                                   fusion_dim=512,
                                                   morgan_bits=2048,
                                                   morgan_dim=None).to(device)

        ckpt_path = None
        if name == 'plqy':
            ckpt_path = r"best_plqy_solute-GIN_solvent-GIN_fusion-concat.pth"
        elif name == 'emi':
            ckpt_path = r'best_emi_solute-GIN_solvent-GIN_fusion-concat.pth'
        elif name == 'abs':
            ckpt_path = r'best_abs_solute-GIN_solvent-GIN_fusion-concat.pth'
        elif name == 'e':
            ckpt_path = r'best_e_solute-GIN_solvent-GIN_fusion-concat.pth'

        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

    @torch.no_grad()
    def forward(self, data):
        data = data.to(device)
        pred = self.model(data).squeeze()

        norm = 2 * ((pred - self.y_min) / (self.y_max - self.y_min)) - 1
        return norm, pred


def calculate_fluro(smiles_list,
                    model_list,
                    morgan_bits=2048,
                    morgan_radius=2):

    scores_plqy, raw_preds_plqy, valid_mask = [], [], []
    scores_emi, raw_preds_emi = [], []
    scores_abs, raw_preds_abs = [], []
    scores_e, raw_preds_e = [], []

    for smi in smiles_list:
        try:
            data = build_data_with_fixed_water(
                solute_smiles=smi,
                morgan_bits=morgan_bits,
                morgan_radius=morgan_radius
            )
            score_plqy, raw_plqy = model_list[0](data)
            score_emi, raw_emi = model_list[1](data)
            score_abs, raw_abs = model_list[2](data)
            score_e, raw_e = model_list[3](data)

            scores_plqy.append(float(score_plqy.detach().cpu().item()))
            scores_emi.append(float(score_emi.detach().cpu().item()))
            scores_abs.append(float(score_abs.detach().cpu().item()))
            scores_e.append(float(score_e.detach().cpu().item()))
            raw_preds_plqy.append(float(raw_plqy.detach().cpu().item()))
            raw_preds_emi.append(float(raw_emi.detach().cpu().item()))
            raw_preds_abs.append(float(raw_abs.detach().cpu().item()))
            raw_preds_e.append(float(raw_e.detach().cpu().item()))
            valid_mask.append(True)
        except Exception as e:
            scores_plqy.append(0.0)
            scores_emi.append(0.0)
            scores_abs.append(0.0)
            scores_e.append(0.0)
            raw_preds_plqy.append(0.0)
            raw_preds_emi.append(0.0)
            raw_preds_abs.append(0.0)
            raw_preds_e.append(0.0)
            valid_mask.append(False)

    return (torch.tensor(scores_plqy, device=device),
            torch.tensor(scores_emi, device=device),
            torch.tensor(scores_abs, device=device),
            torch.tensor(scores_e, device=device),
            torch.tensor(raw_preds_plqy, device=device),
            torch.tensor(raw_preds_emi, device=device),
            torch.tensor(raw_preds_abs, device=device),
            torch.tensor(raw_preds_e, device=device),
            torch.tensor(valid_mask, device=device))








