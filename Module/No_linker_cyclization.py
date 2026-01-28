from rdkit import Chem
from rdkit.Chem import SanitizeFlags


def try_disulfide(peptide_smiles, s1_idx, s2_idx):
    mol = Chem.MolFromSmiles(peptide_smiles)

    mol_h = Chem.AddHs(mol)
    rwmol = Chem.RWMol(mol_h)

    atom1, atom2 = rwmol.GetAtomWithIdx(s1_idx), rwmol.GetAtomWithIdx(s2_idx)
    h_to_remove = []
    for atom in (atom1, atom2):
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == "H":
                h_to_remove.append(nbr.GetIdx())
    h_to_remove = sorted(set(h_to_remove), reverse=True)
    for h_idx in h_to_remove:
        rwmol.RemoveAtom(h_idx)

    rwmol.AddBond(s1_idx, s2_idx, Chem.BondType.SINGLE)
    rwmol.UpdatePropertyCache()
    Chem.SanitizeMol(rwmol)
    mol_noH = Chem.RemoveHs(rwmol)
    final_smiles = Chem.MolToSmiles(mol_noH, canonical=True)
    return final_smiles


def try_amide(peptide_smiles, c_idx, n_idx):
    mol = Chem.MolFromSmiles(peptide_smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    rwmol = Chem.RWMol(mol_h)

    atom_c = rwmol.GetAtomWithIdx(c_idx)
    atom_n = rwmol.GetAtomWithIdx(n_idx)

    if atom_c.GetSymbol() != "C" or atom_n.GetSymbol() != "N":
        return None

    oh_oxygen = None
    for nbr in atom_c.GetNeighbors():
        if nbr.GetSymbol() == "O":
            if any(nn.GetSymbol() == "H" for nn in nbr.GetNeighbors()):
                oh_oxygen = nbr
                break
    if oh_oxygen is None:
        return None

    if rwmol.GetBondBetweenAtoms(c_idx, oh_oxygen.GetIdx()):
        rwmol.RemoveBond(c_idx, oh_oxygen.GetIdx())

    h_to_remove = [nbr.GetIdx() for nbr in oh_oxygen.GetNeighbors() if nbr.GetSymbol() == "H"]
    h_to_remove.append(oh_oxygen.GetIdx())

    for nbr in atom_n.GetNeighbors():
        if nbr.GetSymbol() == "H":
            h_to_remove.append(nbr.GetIdx())
            break

    for h_idx in sorted(set(h_to_remove), reverse=True):
        rwmol.RemoveAtom(h_idx)

    rwmol.AddBond(c_idx, n_idx, Chem.BondType.SINGLE)

    rwmol.UpdatePropertyCache()
    Chem.SanitizeMol(rwmol)

    return Chem.MolToSmiles(Chem.RemoveHs(rwmol), canonical=True)


def try_ester(molecule_smiles, c_idx, o_idx):
    mol = Chem.MolFromSmiles(molecule_smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    rwmol = Chem.RWMol(mol_h)

    atom_c = rwmol.GetAtomWithIdx(c_idx)
    atom_o = rwmol.GetAtomWithIdx(o_idx)

    if atom_c.GetSymbol() != "C" or atom_o.GetSymbol() != "O":
        return None

    carboxyl_oh_oxygen = None
    carbonyl_oxygen = None
    carbon_neighbors = []
    for nbr in atom_c.GetNeighbors():
        if nbr.GetSymbol() == 'O':
            bond = rwmol.GetBondBetweenAtoms(c_idx, nbr.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                carbonyl_oxygen = nbr
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                if any(nh.GetSymbol() == 'H' for nh in nbr.GetNeighbors()):
                    carboxyl_oh_oxygen = nbr
        else:
            carbon_neighbors.append(nbr)

    atoms_to_remove = []

    hydroxyl_h = next((nbr for nbr in atom_o.GetNeighbors() if nbr.GetSymbol() == 'H'), None)
    atoms_to_remove.append(hydroxyl_h.GetIdx())
    carboxyl_h = next((nbr for nbr in carboxyl_oh_oxygen.GetNeighbors() if nbr.GetSymbol() == 'H'), None)
    atoms_to_remove.append(carboxyl_h.GetIdx())
    atoms_to_remove.append(carboxyl_oh_oxygen.GetIdx())
    for idx in sorted(atoms_to_remove, reverse=True):
        rwmol.RemoveAtom(idx)
    new_c_idx = c_idx
    new_o_idx = o_idx
    for idx in sorted(atoms_to_remove):
        if new_c_idx > idx:
            new_c_idx -= 1
        if new_o_idx > idx:
            new_o_idx -= 1

    current_bond_order = 0.0
    atom_c_new = rwmol.GetAtomWithIdx(new_c_idx)
    for bond in rwmol.GetBonds():
        if bond.GetBeginAtomIdx() == new_c_idx or bond.GetEndAtomIdx() == new_c_idx:
            bond_order = bond.GetBondTypeAsDouble()
            current_bond_order += bond_order

    if current_bond_order + 1.0 > 4.0:
        atom_c_new.SetNumExplicitHs(atom_c_new.GetNumExplicitHs() - 1)
    rwmol.AddBond(new_c_idx, new_o_idx, Chem.BondType.SINGLE)
    rwmol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    Chem.SanitizeMol(rwmol)
    return Chem.MolToSmiles(Chem.RemoveHs(rwmol), canonical=True)


def try_thioester(molecule_smiles, c_idx, s_idx):
    mol = Chem.MolFromSmiles(molecule_smiles)

    def remove_free_hydrogens(input_mol):
        rw_mol = Chem.RWMol(input_mol)
        free_h_indices = []
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == "H":
                neighbors = atom.GetNeighbors()
                if not neighbors or all(n.GetSymbol() == "H" for n in neighbors):
                    free_h_indices.append(atom.GetIdx())
        for idx in sorted(free_h_indices, reverse=True):
            rw_mol.RemoveAtom(idx)
        return rw_mol.GetMol(), free_h_indices

    mol_no_free_h, deleted_free_h = remove_free_hydrogens(mol)

    corrected_c_idx = c_idx
    corrected_s_idx = s_idx
    for h_idx in sorted(deleted_free_h):
        if corrected_c_idx > h_idx:
            corrected_c_idx -= 1
        if corrected_s_idx > h_idx:
            corrected_s_idx -= 1

    mol_h = Chem.AddHs(mol_no_free_h)
    rwmol = Chem.RWMol(mol_h)

    atom_c = rwmol.GetAtomWithIdx(corrected_c_idx)
    atom_s = rwmol.GetAtomWithIdx(corrected_s_idx)
    if atom_c.GetSymbol() != "C" or atom_s.GetSymbol() != "S":
        return None

    carboxyl_oh_oxygen = None
    has_carbonyl_oxygen = False
    carbon_non_oxygen_neighbors = []
    for nbr in atom_c.GetNeighbors():
        if nbr.GetSymbol() == "O":
            bond = rwmol.GetBondBetweenAtoms(corrected_c_idx, nbr.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                has_carbonyl_oxygen = True
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                if any(nh.GetSymbol() == "H" for nh in nbr.GetNeighbors()):
                    carboxyl_oh_oxygen = nbr
        else:
            carbon_non_oxygen_neighbors.append(nbr)

    has_thiol_h = any(nbr.GetSymbol() == "H" for nbr in atom_s.GetNeighbors())
    atoms_to_remove = []
    carboxyl_h = next((nbr.GetIdx() for nbr in carboxyl_oh_oxygen.GetNeighbors() if nbr.GetSymbol() == "H"), None)
    atoms_to_remove.append(carboxyl_h)
    atoms_to_remove.append(carboxyl_oh_oxygen.GetIdx())
    thiol_h = next((nbr.GetIdx() for nbr in atom_s.GetNeighbors() if nbr.GetSymbol() == "H"), None)
    atoms_to_remove.append(thiol_h)
    rwmol.RemoveBond(corrected_c_idx, carboxyl_oh_oxygen.GetIdx())
    for idx in sorted(set(atoms_to_remove), reverse=True):
        rwmol.RemoveAtom(idx)
    new_c_idx = corrected_c_idx
    new_s_idx = corrected_s_idx
    for idx in sorted(atoms_to_remove):
        if new_c_idx > idx:
            new_c_idx -= 1
        if new_s_idx > idx:
            new_s_idx -= 1
    if new_c_idx < 0 or new_c_idx >= rwmol.GetNumAtoms():
        return None
    if new_s_idx < 0 or new_s_idx >= rwmol.GetNumAtoms():
        return None
    current_bond_order = 0.0
    for bond in rwmol.GetBonds():
        if bond.GetBeginAtomIdx() == new_c_idx or bond.GetEndAtomIdx() == new_c_idx:
            current_bond_order += bond.GetBondTypeAsDouble()
    if current_bond_order + 1.0 > 4.0:
        atom_c_new = rwmol.GetAtomWithIdx(new_c_idx)
        atom_c_new.SetNumExplicitHs(max(0, atom_c_new.GetNumExplicitHs() - 1))
        current_bond_order -= 1.0

    rwmol.AddBond(new_c_idx, new_s_idx, Chem.BondType.SINGLE)

    rwmol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(rwmol, sanitizeOps=SanitizeFlags.SANITIZE_ADJUSTHS)
    mol_no_excess_h = Chem.RemoveHs(rwmol)
    Chem.SanitizeMol(mol_no_excess_h)

    final_smiles = Chem.MolToSmiles(mol_no_excess_h, canonical=True)
    final_smiles = final_smiles.replace("[SH2]", "S").replace("[S]", "S")
    return final_smiles