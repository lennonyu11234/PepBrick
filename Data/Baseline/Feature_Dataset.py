from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


FEATURE_NAMES = ("morgan", "rdkit", "maccs", "atom_pair", "descriptors")


class MolecularDataset:
    def __init__(self, csv_path, fp_size=2048, radius=2):
        self.name = Path(csv_path).name
        self.frame = pd.read_csv(csv_path)
        self.frame.columns = self.frame.columns.str.strip()
        if not {"SMILES", "Label"}.issubset(self.frame.columns):
            raise ValueError(f"{self.name}: SMILES and Label columns are required.")
        if self.frame.empty or self.frame[["SMILES", "Label"]].isna().any().any():
            raise ValueError(f"{self.name}: empty data or missing SMILES/Label.")
        self.labels = pd.to_numeric(self.frame["Label"]).to_numpy(dtype=float)
        if not np.isfinite(self.labels).all():
            raise ValueError(f"{self.name}: labels must be finite numeric values.")

        self.molecules = []
        self.smiles = []
        for row, value in enumerate(self.frame["SMILES"], start=2):
            mol = Chem.MolFromSmiles(str(value).strip())
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError(f"{self.name}: invalid SMILES at CSV row {row}, SMILES = {value}")
            mol = rdMolStandardize.Cleanup(mol)
            self.molecules.append(mol)
            self.smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        print(f"Loaded {self.name}: {len(self)} records", flush=True)

        self.generators = {
            "morgan": rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=fp_size, includeChirality=True
            ),
            "rdkit": rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size),
            "atom_pair": rdFingerprintGenerator.GetAtomPairGenerator(
                fpSize=fp_size, includeChirality=True
            ),
        }
        self.descriptor_functions = {
            name: function for name, function in Descriptors.descList
            if name not in {"Ipc", "AvgIpc"}
        }
        self.descriptor_names = tuple(sorted(self.descriptor_functions))
        self._features = {}

    def __len__(self):
        return len(self.labels)

    def _encode(self, mol, feature):
        if feature == "descriptors":
            row = np.array([
                self.descriptor_functions[name](mol)
                for name in self.descriptor_names
            ], dtype=float)
            row[~np.isfinite(row)] = np.nan
            return row
        fp = (MACCSkeys.GenMACCSKeys(mol) if feature == "maccs"
              else self.generators[feature].GetFingerprint(mol))
        row = np.zeros(fp.GetNumBits(), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, row)
        return row

    def X(self, feature):
        if feature not in FEATURE_NAMES:
            raise KeyError(f"Choose one of {FEATURE_NAMES}.")
        if feature not in self._features:
            print(f"Computing {feature}: {self.name}", flush=True)
            rows = []
            for index, mol in enumerate(self.molecules, start=1):
                rows.append(self._encode(mol, feature))
                if index % 200 == 0 or index == len(self):
                    print(f"  {feature}: {index}/{len(self)}", flush=True)
            self._features[feature] = np.stack(rows)
        return self._features[feature]

    def __getitem__(self, index):
        return {
            **{
                name: (self._features[name][index] if name in self._features
                       else self._encode(self.molecules[index], name))
                for name in FEATURE_NAMES
            },
            "label": self.labels[index],
            "smiles": self.smiles[index],
        }


def check_split_overlap(train, valid, test):
    splits = {"train": set(train.smiles), "valid": set(valid.smiles), "test": set(test.smiles)}
    for left, right in combinations(splits, 2):
        overlap = splits[left] & splits[right]
        print(f"Molecular overlap {left}/{right}: {len(overlap)}", flush=True)
        if overlap:
            raise ValueError(
                f"{left}/{right}: {len(overlap)} identical molecular structures. "
                "Correct the partitions before training."
            )
