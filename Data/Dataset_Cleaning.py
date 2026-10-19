import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from transformers import AutoTokenizer

input_csv = "data.csv"
output_csv = "cleaned_data.csv"
generator_data = False

tokenizer = AutoTokenizer.from_pretrained(
    "Data/Dataset/SMILES",
    ignore_mismatched_sizes=True
)

df = pd.read_csv(input_csv)
df.columns = df.columns.str.strip()
initial_count = len(df)
amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

if "SMILES" in df.columns:
    source_column = "SMILES"
    sequence_input = False
elif "SEQUENCE" in df.columns:
    source_column = "SEQUENCE"
    sequence_input = True
else:
    raise ValueError("The CSV must contain SMILES or SEQUENCE.")

def standardize(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    if sequence_input:
        if not set(value).issubset(amino_acids):
            return None
        mol = Chem.MolFromSequence(value)
    else:
        mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    return Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True
    )

def is_uncharged(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return all(atom.GetFormalCharge() == 0 for atom in mol.GetAtoms())

def token_count(smiles):
    return len(tokenizer.encode(
        smiles,
        add_special_tokens=True,
        truncation=False
    ))

df["SMILES"] = df[source_column].map(standardize)
df = df.dropna(subset=["SMILES"]).copy()
valid_count = len(df)

df = df.drop_duplicates(subset=["SMILES"] if generator_data else None)
unique_count = len(df)

if generator_data:
    df = df[df["SMILES"].map(is_uncharged)].copy()
    uncharged_count = len(df)
    df = df[df["SMILES"].map(token_count) <= 100].copy()

df.to_csv(output_csv, index=False)

print(f"Initial records: {initial_count}")
print(f"Invalid or unsupported records removed: {initial_count - valid_count}")
print(f"Duplicates removed: {valid_count - unique_count}")

if generator_data:
    print(f"Charged molecules removed: {unique_count - uncharged_count}")
    print(f"Overlength molecules removed: {uncharged_count - len(df)}")

print(f"Final records: {len(df)}")