from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import torch
import numpy as np
from Module.Linker_policy import CyclicPolicy
from Module.No_linker_cyclization import try_ester, try_disulfide, try_amide, try_thioester
from openbabel import pybel
from Module.Linker_cyclization import connect_linker_pep

SMARTS = {
    'thiol': '[#16;X2;H1]',
    'prim_amine': '[N;!$(N-C(=O))]-[C]-C(=O)',
    'hydroxyl': '[OX2H;!$(OC(=O))]',
    'carboxyl': 'C(=O)[OH]'
}

ACTION_TYPES = ['disulfide', 'amide', 'ester', 'thioester', 'none', 'linker']

TYPE_TO_ONEHOT = {
    'thiol': 0,
    'prim_amine': 1,
    'hydroxyl': 2,
    'carboxyl': 3
}


def add_explicit_hydrogens(smiles):
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    smiles_with_h = mol.write("smi", opt={"h": True, "can": True}).strip()
    smiles_with_h = smiles_with_h.replace("+", "").replace("-", "")
    return smiles_with_h


def find_sites(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sites = []
    for t, patt in SMARTS.items():
        mp = Chem.MolFromSmarts(patt)
        matches = mol.GetSubstructMatches(mp)
        for match in matches:
            atom_idx = match[0]
            sites.append({'type': t, 'atom_idx': atom_idx})
    return mol, sites


def build_site_feature_matrix(mol, sites, max_sites=32):
    n_sites = min(len(sites), max_sites)
    feat_dim = 4 + 1 + 1
    mat = np.zeros((1, max_sites, feat_dim), dtype=np.float32)
    for i in range(n_sites):
        s = sites[i]
        t = s['type']
        idx = s['atom_idx']
        onehot = np.zeros(4, dtype=np.float32)
        onehot[TYPE_TO_ONEHOT[t]] = 1.0
        atom = mol.GetAtomWithIdx(idx)
        degree = float(atom.GetDegree())
        norm_idx = float(idx) / max(1, mol.GetNumAtoms() - 1)
        mat[0, i, :4] = onehot
        mat[0, i, 4] = degree
        mat[0, i, 5] = norm_idx
    return torch.from_numpy(mat)


def mol_to_fp_tensor(mol, n_bits=2048, radius=2):
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1, n_bits), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr[0])
    return torch.from_numpy(arr)


def is_action_applicable(site_i, site_j, action_type):
    ti = site_i['type']
    tj = site_j['type']
    if action_type == 'disulfide':
        return ti == 'thiol' and tj == 'thiol' and site_i['atom_idx'] != site_j['atom_idx']
    if action_type == 'amide':
        return (ti == 'carboxyl' and tj == 'prim_amine') or (tj == 'carboxyl' and ti == 'prim_amine')
    if action_type == 'ester':
        return (ti == 'carboxyl' and tj == 'hydroxyl') or (tj == 'carboxyl' and ti == 'hydroxyl')
    if action_type == 'thioester':
        return (ti == 'carboxyl' and tj == 'thiol') or (tj == 'thiol' and ti == 'carboxyl')
    if action_type == 'none' or 'linker':
        return True
    return False


def sample_action_from_logits(pair_logits, sites, mol, temperature=1.0):
    logits = pair_logits.detach().cpu().numpy()[0]  # (n, n, n_types)
    n = logits.shape[0]
    mask = np.zeros_like(logits, dtype=bool)
    for i in range(n):
        for j in range(n):
            for t_idx, atype in enumerate(ACTION_TYPES):
                if i == j:
                    mask[i, j, t_idx] = True
                    continue
                if i >= len(sites) or j >= len(sites):
                    mask[i, j, t_idx] = True
                    continue
                if not is_action_applicable(sites[i], sites[j], atype):
                    mask[i, j, t_idx] = True
    logits_masked = np.copy(logits)
    logits_masked[mask] = -1e9
    flat = logits_masked.reshape(-1)
    probs = np.exp(flat / max(temperature, 1e-6))
    probs = probs / (probs.sum() + 1e-12)
    idx = np.random.choice(len(probs), p=probs)
    t_idx = idx % len(ACTION_TYPES)
    j_idx = (idx // len(ACTION_TYPES)) % n
    i_idx = (idx // len(ACTION_TYPES)) // n
    return int(i_idx), int(j_idx), int(t_idx)

