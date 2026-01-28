import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Data/Dataset/SMILES",
    ignore_mismatched_sizes=False
)


class SMILESPretrainDataset(Dataset):
    def __init__(self, path, max_len=None, mask_prob=0.15):
        self.smiles_list = self.read_csv(path)
        self.mask_prob = mask_prob
        self.mask_token = tokenizer.mask_token
        self.pad_token = tokenizer.pad_token
        self.vocab = tokenizer.vocab
        self.max_len = max_len if max_len is not None else self.get_max_length()
        self.tokenized_sequences = self.pre_tokenize()

    def __len__(self):
        return len(self.smiles_list)

    def read_csv(self, path):
        df = pd.read_csv(path)
        smiles_list = [str(s).strip() for s in df['SMILES'].tolist() if pd.notna(s) and str(s).strip() != '']
        return smiles_list

    def get_max_length(self):
        max_len = 0
        for smiles in self.smiles_list:
            tokens = tokenizer.tokenize(smiles)
            if len(tokens) > max_len:
                max_len = len(tokens)
        return max_len

    def pre_tokenize(self):
        tokenized = []
        for smiles in self.smiles_list:
            tokens = tokenizer.tokenize(smiles)
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            else:
                tokens += [self.pad_token] * (self.max_len - len(tokens))
            tokenized.append(tokens)
        return tokenized

    def _mask_tokens(self, tokens):
        input_tokens = tokens.copy()
        target_ids = [-100] * len(tokens)

        for i in range(len(tokens)):
            if tokens[i] == self.pad_token:
                continue
            if random.random() < self.mask_prob:
                if random.random() < 0.8:
                    input_tokens[i] = self.mask_token
                elif random.random() < 0.5:
                    input_tokens[i] = random.choice(list(self.vocab.keys()))
                target_ids[i] = self.vocab[tokens[i]]

        return input_tokens, target_ids

    def __getitem__(self, idx):
        tokens = self.tokenized_sequences[idx]
        input_tokens, target_tokens = self._mask_tokens(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1 if t != self.pad_token else 0 for t in tokens]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'target_ids': torch.tensor(target_tokens, dtype=torch.long)
        }

