from rdkit import Chem
from openbabel import pybel


def add_explicit_hydrogens(smiles):
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    smiles_with_h = mol.write("smi", opt={"h": True, "can": True}).strip()
    smiles_with_h = smiles_with_h.replace("+", "").replace("-", "")
    return smiles_with_h


def determine_atom_type(atom):
    if atom.GetSymbol() == 'C':
        return 'carboxyl'
    elif atom.GetSymbol() == 'S':
        return 'thiol'
    elif atom.GetSymbol() == 'N':
        return 'prim_amine'
    elif atom.GetSymbol() == 'O':
        return 'hydroxyl'


def determine_link_type(atom1_type, atom2_type):
    if atom1_type == 'thiol' and atom2_type == 'thiol':
        return 'disulfide'
    elif atom1_type == 'prim_amine' and atom2_type == 'prim_amine':
        return 'n_n'
    elif atom1_type == 'hydroxyl' and atom2_type == 'hydroxyl':
        return 'o_o'
    elif (atom1_type == 'prim_amine' and atom2_type == 'carboxyl') or (atom1_type == 'carboxyl' and atom2_type == 'prim_amine'):
        return 'amide'
    elif (atom1_type == 'hydroxyl' and atom2_type == 'carboxyl') or (atom1_type == 'carboxyl' and atom2_type == 'hydroxyl'):
        return 'ester'
    elif (atom1_type == 'thiol' and atom2_type == 'carboxyl') or (atom1_type == 'carboxyl' and atom2_type == 'thiol'):
        return 'thioester'
    elif (atom1_type == 'thiol' and atom2_type == 'prim_amine') or (atom1_type == 'prim_amine' and atom2_type == 'thiol'):
        return 'n_s'
    elif (atom1_type == 'thiol' and atom2_type == 'hydroxyl') or (atom1_type == 'hydroxyl' and atom2_type == 'thiol'):
        return 's_o'
    elif (atom1_type == 'prim_amine' and atom2_type == 'hydroxyl') or (atom1_type == 'hydroxyl' and atom2_type == 'prim_amine'):
        return 'n_o'
    elif (atom1_type == 'carboxyl' and atom2_type == 'carboxyl'):
        return 'c_c'


def connect_s_and_br(mol, s_idx, br_idx):
    emol = Chem.EditableMol(mol)
    br_atom = mol.GetAtomWithIdx(br_idx)
    carbon = next((n for n in br_atom.GetNeighbors() if n.GetAtomicNum() == 6), None)
    c_idx = carbon.GetIdx()
    emol.AddBond(s_idx, c_idx, Chem.BondType.SINGLE)
    deleted_count = 0

    if br_idx < s_idx:
        s_idx -= 1
    emol.RemoveAtom(br_idx)
    deleted_count += 1
    new_mol = emol.GetMol()
    s_atom = new_mol.GetAtomWithIdx(s_idx)
    hs = [n.GetIdx() for n in s_atom.GetNeighbors() if n.GetAtomicNum() == 1]
    for h in sorted(hs, reverse=True):
        if h < s_idx:
            s_idx -= 1
        emol.RemoveAtom(h)
        deleted_count += 1
    new_mol = emol.GetMol()
    return new_mol, s_idx, c_idx, deleted_count


def connect_n_and_br(mol, n_idx, br_idx):
    emol = Chem.EditableMol(mol)
    br_atom = mol.GetAtomWithIdx(br_idx)
    carbon = next((n for n in br_atom.GetNeighbors() if n.GetAtomicNum() == 6), None)
    c_idx = carbon.GetIdx()
    emol.AddBond(n_idx, c_idx, Chem.BondType.SINGLE)
    deleted_count = 0
    if br_idx < n_idx:
        n_idx -= 1
    emol.RemoveAtom(br_idx)
    deleted_count += 1
    new_mol = emol.GetMol()
    n_atom = new_mol.GetAtomWithIdx(n_idx)
    hs = [n.GetIdx() for n in n_atom.GetNeighbors() if n.GetAtomicNum() == 1]
    if hs:
        h = hs[0]
        if h < n_idx:
            n_idx -= 1
        emol.RemoveAtom(h)
        deleted_count += 1
    new_mol = emol.GetMol()
    return new_mol, n_idx, c_idx, deleted_count


def connect_o_and_br(mol, o_idx, br_idx):
    emol = Chem.EditableMol(mol)
    br_atom = mol.GetAtomWithIdx(br_idx)
    carbon = next((n for n in br_atom.GetNeighbors() if n.GetAtomicNum() == 6), None)
    c_idx = carbon.GetIdx()
    emol.AddBond(o_idx, c_idx, Chem.BondType.SINGLE)

    deleted_count = 0
    if br_idx < o_idx:
        o_idx -= 1
    emol.RemoveAtom(br_idx)
    deleted_count += 1
    new_mol = emol.GetMol()
    n_atom = new_mol.GetAtomWithIdx(o_idx)
    hs = [n.GetIdx() for n in n_atom.GetNeighbors() if n.GetAtomicNum() == 1]
    if hs:
        h = hs[0]
        if h < o_idx:
            o_idx -= 1
        emol.RemoveAtom(h)
        deleted_count += 1
    new_mol = emol.GetMol()
    return new_mol, o_idx, c_idx, deleted_count


def connect_c_and_br(mol, c_idx, br_idx):
    emol = Chem.EditableMol(mol)
    br_atom = mol.GetAtomWithIdx(br_idx)
    carbon = next((n for n in br_atom.GetNeighbors() if n.GetAtomicNum() == 6), None)
    neibor_c_idx = carbon.GetIdx()
    deleted_atoms = []
    deleted_atoms.append(br_idx)
    if br_idx < c_idx:
        c_idx -= 1
    if br_idx < neibor_c_idx:
        neibor_c_idx -= 1
    emol.RemoveAtom(br_idx)
    c_atom = emol.GetMol().GetAtomWithIdx(c_idx)
    oxygens = [n.GetIdx() for n in c_atom.GetNeighbors() if n.GetAtomicNum() == 8]
    hydroxyl_oxygen = None
    for o_idx in oxygens:
        o_atom = emol.GetMol().GetAtomWithIdx(o_idx)
        has_hydrogen = any(n.GetAtomicNum() == 1 for n in o_atom.GetNeighbors())
        if has_hydrogen:
            hydroxyl_oxygen = o_idx
            break
    o_atom = emol.GetMol().GetAtomWithIdx(hydroxyl_oxygen)
    hydrogens = [n.GetIdx() for n in o_atom.GetNeighbors() if n.GetAtomicNum() == 1]
    for h_idx in sorted(hydrogens, reverse=True):
        deleted_atoms.append(h_idx)
        if h_idx < c_idx:
            c_idx -= 1
        if h_idx < neibor_c_idx:
            neibor_c_idx -= 1
        emol.RemoveAtom(h_idx)
    deleted_atoms.append(hydroxyl_oxygen)
    if hydroxyl_oxygen < c_idx:
        c_idx -= 1
    if hydroxyl_oxygen < neibor_c_idx:
        neibor_c_idx -= 1
    emol.RemoveAtom(hydroxyl_oxygen)
    emol.AddBond(c_idx, neibor_c_idx, Chem.BondType.SINGLE)
    new_mol = emol.GetMol()
    return new_mol, c_idx, neibor_c_idx, deleted_atoms


def connect_linker_pep(pep_smiles, linker_smiles, atom_idx1, atom_idx2):
    try:
        pep_smiles_with_h = add_explicit_hydrogens(pep_smiles)

        pep_mol = Chem.MolFromSmiles(pep_smiles_with_h)
        lin_mol = Chem.MolFromSmiles(linker_smiles)

        pep_mol = Chem.AddHs(pep_mol)
        lin_mol = Chem.AddHs(lin_mol)
        pep_mol = Chem.RWMol(pep_mol)
        lin_mol = Chem.RWMol(lin_mol)
        br_pattern = Chem.MolFromSmarts("[Br]")
        br_matches = lin_mol.GetSubstructMatches(br_pattern)

        br1_idx, br2_idx = [match[0] for match in br_matches[:2]]

        atom1 = pep_mol.GetAtomWithIdx(atom_idx1)
        atom2 = pep_mol.GetAtomWithIdx(atom_idx2)
        atom1_type = determine_atom_type(atom1)
        atom2_type = determine_atom_type(atom2)

        combined = Chem.CombineMols(pep_mol, lin_mol)
        peptide_num_atoms = pep_mol.GetNumAtoms()

        combined_br1 = peptide_num_atoms + br1_idx
        combined_br2 = peptide_num_atoms + br2_idx

        link_type = determine_link_type(atom1_type, atom2_type)

        if link_type == 'disulfide':
            intermediate, s1_new_idx, c1_idx, deleted1 = connect_s_and_br(combined, atom_idx1, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, s2_new_idx, c2_idx, _ = connect_s_and_br(intermediate, atom_idx2, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)

        elif link_type == 'amide':
            if atom1_type == 'prim_amine':
                n_idx, c_idx = atom_idx1, atom_idx2
            else:
                n_idx, c_idx = atom_idx2, atom_idx1

            intermediate, n1_new_idx, c1_idx, deleted1 = connect_n_and_br(combined, n_idx, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, c2_new_idx, c3_idx, _ = connect_c_and_br(intermediate, c_idx, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)

        elif link_type == 'thioester':
            if atom1_type == 'thiol':
                s_idx, c_idx = atom_idx1, atom_idx2
            else:
                s_idx, c_idx = atom_idx2, atom_idx1

            intermediate, s1_new_idx, c1_idx, deleted1 = connect_n_and_br(combined, s_idx, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, c2_new_idx, c3_idx, _ = connect_c_and_br(intermediate, c_idx, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)

        elif link_type == 'ester':
            if atom1_type == 'hydroxyl':
                o_idx, c_idx = atom_idx1, atom_idx2
            else:
                o_idx, c_idx = atom_idx2, atom_idx1

            intermediate, s1_new_idx, o1_idx, deleted1 = connect_o_and_br(combined, o_idx, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, c2_new_idx, c3_idx, _ = connect_c_and_br(intermediate, c_idx, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)

        elif link_type == 's_o':
            if atom1_type == 'thiol':
                s_idx, o_idx = atom_idx1, atom_idx2
            else:
                s_idx, o_idx = atom_idx2, atom_idx1

            intermediate, s1_new_idx, o1_idx, deleted1 = connect_s_and_br(combined, s_idx, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, c2_new_idx, c3_idx, _ = connect_o_and_br(intermediate, o_idx, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)

        elif link_type == 'n_o':
            if atom1_type == 'prim_amine':
                n_idx, o_idx = atom_idx1, atom_idx2
            else:
                n_idx, o_idx = atom_idx2, atom_idx1

            intermediate, s1_new_idx, o1_idx, deleted1 = connect_n_and_br(combined, n_idx, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, c2_new_idx, c3_idx, _ = connect_o_and_br(intermediate, o_idx, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)

        elif link_type == 'n_s':
            if atom1_type == 'prim_amine':
                n_idx, s_idx = atom_idx1, atom_idx2
            else:
                n_idx, s_idx = atom_idx2, atom_idx1

            intermediate, s1_new_idx, o1_idx, deleted1 = connect_n_and_br(combined, n_idx, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, c2_new_idx, c3_idx, _ = connect_s_and_br(intermediate, s_idx, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)

        elif link_type == 'n_n':
            intermediate, s1_new_idx, o1_idx, deleted1 = connect_n_and_br(combined, atom_idx1, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, c2_new_idx, c3_idx, _ = connect_n_and_br(intermediate, atom_idx2, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)

        elif link_type == 'o_o':
            intermediate, s1_new_idx, o1_idx, deleted1 = connect_o_and_br(combined, atom_idx1, combined_br1)
            br2_new_idx = combined_br2
            if combined_br2 > combined_br1:
                br2_new_idx -= deleted1
            final_mol, c2_new_idx, c3_idx, _ = connect_o_and_br(intermediate, atom_idx2, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)


        elif link_type == 'c_c':
            intermediate, s1_new_idx, o1_idx, deleted_atoms1 = connect_c_and_br(combined, atom_idx1, combined_br1)
            br2_adjust = sum(1 for idx in deleted_atoms1 if idx < combined_br2)
            br2_new_idx = combined_br2 - br2_adjust
            atom2_adjust = sum(1 for idx in deleted_atoms1 if idx < atom_idx2)
            atom_idx2_new = atom_idx2 - atom2_adjust
            final_mol, c2_new_idx, c3_idx, _ = connect_c_and_br(intermediate, atom_idx2_new, br2_new_idx)
            final_mol_no_h = Chem.RemoveHs(final_mol)
            return Chem.MolToSmiles(final_mol_no_h)
    except:
        return None
