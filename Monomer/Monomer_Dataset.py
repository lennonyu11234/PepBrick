import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Data/Dataset/SMILES", ignore_mismatched_sizes=False)


class AASmilesDataset(Dataset):
    def __init__(self, path, mode='regression'):
        self.aa_smiles_list, self.labels = self.read_csv(path)
        self.max_num_aa, self.max_smiles_len = self.get_max_dims()
        self.max_smiles_len = 100
        self.padded_matrices = self.pad_matrices()
        self.mode = mode

    def __len__(self):
        return len(self.aa_smiles_list)

    def __getitem__(self, idx):
        token_matrix = self.padded_matrices[idx]
        id_matrix = [tokenizer.convert_tokens_to_ids(tokens) for tokens in token_matrix]
        input_ids = torch.tensor(id_matrix, dtype=torch.long)
        attention_mask = (input_ids != tokenizer.vocab[tokenizer.pad_token]).long()

        label = self.labels[idx]
        if self.mode == 'classification':
            label = torch.tensor(1 if label >= -6 else 0, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

    def read_csv(self, path):
        df = pd.read_csv(path)
        labels = df['Permeability'].tolist()
        aa_smiles_list = []
        aa_smiles_cols = [f'AA{i}_SMILES' for i in range(1, 6)]
        for _, row in df.iterrows():
            sample_smiles = []
            for col in aa_smiles_cols:
                smiles = row[col]
                if pd.notna(smiles) and str(smiles).strip() != '':
                    sample_smiles.append(str(smiles).strip())
            aa_smiles_list.append(sample_smiles)
        return aa_smiles_list, labels

    def get_max_dims(self):
        max_num_aa = max(len(smiles_list) for smiles_list in self.aa_smiles_list)
        max_smiles_len = 0
        for smiles_list in self.aa_smiles_list:
            for smiles in smiles_list:
                tokens = tokenizer.tokenize(smiles)
                if len(tokens) > max_smiles_len:
                    max_smiles_len = len(tokens)
        return max_num_aa, max_smiles_len

    def pad_matrices(self):
        padded_matrices = []
        for smiles_list in self.aa_smiles_list:
            token_rows = []
            for smiles in smiles_list:
                tokens = tokenizer.tokenize(smiles)
                if len(tokens) < self.max_smiles_len:
                    tokens += [tokenizer.pad_token] * (self.max_smiles_len - len(tokens))
                token_rows.append(tokens)
            num_rows = len(token_rows)
            if num_rows < self.max_num_aa:
                pad_row = [tokenizer.pad_token] * self.max_smiles_len
                token_rows += [pad_row] * (self.max_num_aa - num_rows)

            padded_matrices.append(token_rows)
        return padded_matrices


class AASmilesDatasetPre(Dataset):
    def __init__(self, path, mode='regression'):
        self.aa_smiles_list = self.read_csv(path)
        self.max_num_aa, self.max_smiles_len = self.get_max_dims()
        self.max_smiles_len = 100
        self.padded_matrices = self.pad_matrices()
        self.mode = mode

    def __len__(self):
        return len(self.aa_smiles_list)

    def __getitem__(self, idx):
        token_matrix = self.padded_matrices[idx]
        id_matrix = [tokenizer.convert_tokens_to_ids(tokens) for tokens in token_matrix]
        input_ids = torch.tensor(id_matrix, dtype=torch.long)
        attention_mask = (input_ids != tokenizer.vocab[tokenizer.pad_token]).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def read_csv(self, path):
        df = pd.read_csv(path)
        aa_smiles_list = []
        aa_smiles_cols = [f'NNAA_{i}' for i in range(1, 6)]
        for _, row in df.iterrows():
            sample_smiles = []
            for col in aa_smiles_cols:
                smiles = row[col]
                if pd.notna(smiles) and str(smiles).strip() != '':
                    sample_smiles.append(str(smiles).strip())
            aa_smiles_list.append(sample_smiles)
        return aa_smiles_list

    def get_max_dims(self):
        max_num_aa = max(len(smiles_list) for smiles_list in self.aa_smiles_list)
        max_smiles_len = 0
        for smiles_list in self.aa_smiles_list:
            for smiles in smiles_list:
                tokens = tokenizer.tokenize(smiles)
                if len(tokens) > max_smiles_len:
                    max_smiles_len = len(tokens)
        return max_num_aa, max_smiles_len

    def pad_matrices(self):
        padded_matrices = []
        for smiles_list in self.aa_smiles_list:
            token_rows = []
            for smiles in smiles_list:
                tokens = tokenizer.tokenize(smiles)
                if len(tokens) < self.max_smiles_len:
                    tokens += [tokenizer.pad_token] * (self.max_smiles_len - len(tokens))
                token_rows.append(tokens)
            num_rows = len(token_rows)
            if num_rows < self.max_num_aa:
                pad_row = [tokenizer.pad_token] * self.max_smiles_len
                token_rows += [pad_row] * (self.max_num_aa - num_rows)

            padded_matrices.append(token_rows)
        return padded_matrices
