import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
tokenizer = AutoTokenizer.from_pretrained("Data/Dataset/SMILES",
                                          ignore_mismatched_sizes=True)


def idx_to_seqs(input_ids_batch):
    ids = input_ids_batch.detach().cpu().tolist()
    seqs = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [s.replace(" ", "") for s in seqs]


class DatasetSMILES(Dataset):
    def __init__(self, path):
        self.sequences = self.read_csv(path)
        self.max_len = self.get_max_length(self.sequences)
        self.padded_sequences = self.pad_sequence(self.sequences, self.max_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.padded_sequences[idx]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = (input_ids != tokenizer.vocab[tokenizer.pad_token]).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def read_csv(self, path):
        df = pd.read_csv(path)
        sequences = df['SMILES'].tolist()
        return sequences

    def get_max_length(self, sequences):
        max_len = 0
        for seq in sequences:
            tokens = tokenizer(seq, return_tensors='pt')
            input_ids = tokens['input_ids'].squeeze(dim=0)
            if len(input_ids) > max_len:
                max_len = len(input_ids)
        return max_len

    def pad_sequence(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            tokens = tokenizer.tokenize(seq)
            if len(tokens) < max_length:
                tokens += [tokenizer.pad_token] * (max_length - len(tokens))
            padded_sequences.append(tokens)
        return padded_sequences


class DatasetSMILESFast(Dataset):
    def __init__(
        self,
        csv_path: str,
        smiles_col: str = "SMILES",
        max_length: int = 70,
        truncation: bool = True,
        cache_path: str = None,
        chunk_size: int = 50000,
        dtype_store: torch.dtype = torch.int32,
    ):
        self.max_len = max_length

        if cache_path is not None and os.path.exists(cache_path):
            pack = torch.load(cache_path, map_location="cpu")
            self.input_ids = pack["input_ids"]
            self.attention_mask = pack["attention_mask"]
            self.max_len = int(pack.get("max_len", self.input_ids.size(1)))
            return

        df = pd.read_csv(csv_path)
        smiles_list = df[smiles_col].astype(str).tolist()
        n = len(smiles_list)

        input_ids_chunks = []
        attn_chunks = []

        for start in range(0, n, chunk_size):
            batch_smiles = smiles_list[start:start + chunk_size]
            enc = tokenizer(
                batch_smiles,
                add_special_tokens=True,
                padding="max_length",
                truncation=truncation,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids_chunks.append(enc["input_ids"].to(dtype_store))
            attn_chunks.append(enc["attention_mask"].to(torch.uint8))  # mask 用 uint8 更省

        self.input_ids = torch.cat(input_ids_chunks, dim=0)         # [N, L]
        self.attention_mask = torch.cat(attn_chunks, dim=0)         # [N, L]

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(
                {
                    "input_ids": self.input_ids,
                    "attention_mask": self.attention_mask,
                    "max_len": self.max_len
                },
                cache_path
            )

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }
























