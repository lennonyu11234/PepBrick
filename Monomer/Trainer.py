import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import csv
from tqdm import tqdm
from sklearn.metrics import r2_score
from args import args

from Monomer.Module import AASmilesGRUModel
from Monomer.Module_test import ImprovedAASmilesModel
from Monomer.Monomer_Dataset import AASmilesDataset, tokenizer


class Trainer:
    def __init__(self, data_path, save_path, tokenizer, log_file, device=None):
        self.device = torch.device(device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.dataset = AASmilesDataset(path=data_path, mode='regression')
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=min(os.cpu_count(), 32),
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=min(os.cpu_count(), 32),
            persistent_workers=True
        )
        self.model = AASmilesGRUModel(tokenizer).to(self.device)

        self.model.embedding.load_state_dict(torch.load('Data\Model\smiles_pretrained_embedding.pth'))
        self.model.smiles_gru.load_state_dict(torch.load('Data\Model\smiles_pretrained_gru.pth'))

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_main_loss = 0.0
        total_aux_loss = 0.0
        all_preds = []
        all_labels = []
        train_pbar = tqdm(self.train_loader, desc="Training")

        for batch in train_pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            pred, contributions, aux_loss = self.model(input_ids, attention_mask, target=labels)
            main_loss = self.criterion(pred, labels)
            total_loss = main_loss + aux_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            batch_size = input_ids.size(0)
            total_main_loss += main_loss.item() * batch_size
            total_aux_loss += aux_loss.item() * batch_size
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            train_pbar.set_postfix({
                "main_loss": f"{main_loss.item():.4f}",
                "aux_loss": f"{aux_loss.item():.4f}",
                "total_loss": f"{total_loss.item():.4f}"
            })
        train_avg_main_loss = total_main_loss / len(self.train_dataset)
        train_avg_aux_loss = total_aux_loss / len(self.train_dataset)
        train_r2 = r2_score(np.concatenate(all_labels), np.concatenate(all_preds))
        return train_avg_main_loss, train_avg_aux_loss, train_r2

    def val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_contributions = []
        val_pbar = tqdm(self.val_loader, desc="Validating")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)
                pred, contributions, _ = self.model(input_ids, attention_mask)
                loss = self.criterion(pred, labels)
                total_loss += loss.item() * input_ids.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                aa_mask = (attention_mask.sum(dim=2) > 0).float().cpu().numpy()
                contrib_np = contributions.cpu().numpy()
                for b_idx in range(contrib_np.shape[0]):
                    valid_mask = aa_mask[b_idx] == 1
                    valid_contrib = contrib_np[b_idx, valid_mask]
                    all_contributions.extend(valid_contrib.tolist())
                val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        val_avg_loss = total_loss / len(self.val_dataset)
        val_r2 = r2_score(np.concatenate(all_labels), np.concatenate(all_preds))
        if len(all_contributions) == 0:
            contrib_stats = {
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0
            }
        else:
            contrib_arr = np.array(all_contributions)
            contrib_stats = {
                "mean": float(contrib_arr.mean()),
                "std": float(contrib_arr.std()),
                "min": float(contrib_arr.min()),
                "max": float(contrib_arr.max()),
                "count": len(contrib_arr)
            }
        return val_avg_loss, val_r2, contrib_stats

    def save_best_model(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Best model saved to {self.save_path} (Val Loss: {self.best_val_loss:.4f})")

    def train(self, epochs=args.epochs):
        write_header = not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0
        for epoch in range(1, epochs + 1):
            train_main_loss, train_aux_loss, train_r2 = self.train_epoch()
            val_loss, val_r2, contrib_stats = self.val_epoch()
            self.scheduler.step(val_loss)
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train: Main Loss = {train_main_loss:.4f}, Aux Loss = {train_aux_loss:.4f}, R2 = {train_r2:.4f}")
            print(f"Val: Loss = {val_loss:.4f}, R2 = {val_r2:.4f}")
            print(f"Contributions Stats: Mean = {contrib_stats['mean']:.4f}, Std = {contrib_stats['std']:.4f}, "
                  f"Range = [{contrib_stats['min']:.4f}, {contrib_stats['max']:.4f}], Count = {contrib_stats['count']}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_best_model()

            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'epoch', 'train_main_loss', 'train_aux_loss', 'train_r2',
                        'val_loss', 'val_r2', 'contrib_mean', 'contrib_std', 'contrib_min', 'contrib_max'
                    ])
                    write_header = False
                writer.writerow([
                    epoch,
                    round(train_main_loss, 4),
                    round(train_aux_loss, 4),
                    round(train_r2, 4),
                    round(val_loss, 4),
                    round(val_r2, 4),
                    round(contrib_stats['mean'], 4),
                    round(contrib_stats['std'], 4),
                    round(contrib_stats['min'], 4),
                    round(contrib_stats['max'], 4),
                ])

        print(f"\nTraining finished. Best Val Loss: {self.best_val_loss:.4f}")



if __name__ == '__main__':
    data_path = r'Data\Dataset/AA_Cyclic.csv'
    save_path = "Data/Model/test.pth"
    log_file = 'Data/Result/1.12/cyc.csv'
    trainer = Trainer(
        data_path=data_path,
        save_path=save_path,
        tokenizer=tokenizer,
        log_file=log_file
    )
    trainer.train()

