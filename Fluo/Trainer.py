import os
import csv
from tqdm import tqdm
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch.nn.functional as F

from Dataset import SoluteSolventDataset, args, set_seed
from fluorophore.FLSF.Fluore_Module import GraphSolventScaffoldRegressor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self,
                 train_csv_path: str, valid_csv_path: str,
                 train_root_dir: str, valid_root_dir: str,
                 save_dir: str,
                 log_file: str):
        self.train_dataset = SoluteSolventDataset(root=train_root_dir, csv_path=train_csv_path)
        self.valid_dataset = SoluteSolventDataset(root=valid_root_dir, csv_path=valid_csv_path)

        self.train_loader = GeoDataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True,
                                          follow_batch=['s_x'])
        self.valid_loader = GeoDataLoader(self.valid_dataset, batch_size=args.batch_size, shuffle=False,
                                          follow_batch=['s_x'])

        self.model = GraphSolventScaffoldRegressor(
            emb_dim=args.emb_dim,
            num_layer=args.num_layer,
            drop_ratio=args.drop_ratio,
            JK=args.JK,
            pooling=args.pooling,
            solute_gnn=args.solute_gnn,
            solvent_gnn=args.solvent_gnn,
            use_scaffold=args.use_scaffold,
            scaffold_vocab=args.scaffold_vocab,
            scaffold_emb=args.scaffold_emb,
            fusion_mode=args.fusion_mode,
            fusion_dim=args.fusion_dim
        ).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file = log_file

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for data in self.train_loader:
            data = data.to(device)
            y = data.label.view(-1).float()

            self.optimizer.zero_grad()
            pred = self.model(data).view(-1)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

        train_r2 = r2_score(all_labels, all_preds) if len(all_labels) > 1 else 0.0
        return total_loss, train_r2

    def val_epoch(self):
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        n_samples = 0

        all_preds, all_labels = [], []

        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(device)
                y = data.label.view(-1).float()  # [B]
                pred = self.model(data).view(-1).float()  # [B]

                mse = F.mse_loss(pred, y, reduction="sum").item()
                mae = F.l1_loss(pred, y, reduction="sum").item()

                total_mse += mse
                total_mae += mae
                n_samples += y.numel()

                all_preds.extend(pred.detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())

        mse_mean = total_mse / max(n_samples, 1)
        mae_mean = total_mae / max(n_samples, 1)
        rmse_mean = (mse_mean ** 0.5)

        val_r2 = r2_score(all_labels, all_preds) if len(all_labels) > 1 else 0.0
        return mae_mean, mse_mean, rmse_mean, val_r2

    def save_best_model(self):
        name = f"best_plqy-{args.solute_gnn}.pth"
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, name))

    def train(self, epochs: int = args.epochs):
        best_r2 = -1e9

        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        write_header = (not os.path.exists(self.log_file)) or (os.path.getsize(self.log_file) == 0)

        for epoch in tqdm(range(1, epochs + 1), total=epochs, desc="Training"):
            train_loss, train_r2 = self.train_epoch()
            val_mae, val_mse, val_rmse, val_r2 = self.val_epoch()

            print(
                f"\ntrain loss:{train_loss:.4f} | train R2:{train_r2:.4f} | "
                f"val MAE:{val_mae:.4f} | val MSE:{val_mse:.4f} | val RMSE:{val_rmse:.4f} | val R2:{val_r2:.4f}"
            )

            if val_r2 >= best_r2:
                best_r2 = val_r2
                self.save_best_model()

            with open(self.log_file, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["epoch", "train_loss", "train_r2", "val_mae", "val_mse", "val_rmse", "val_r2"])
                    write_header = False
                w.writerow([
                    epoch,
                    round(train_loss, 6),
                    round(train_r2, 6),
                    round(val_mae, 6),
                    round(val_mse, 6),
                    round(val_rmse, 6),
                    round(val_r2, 6),
                ])


if __name__ == "__main__":
    set_seed(args.seed)
    train_path, train_root = '../Data/Dataset/abs_train.csv', '../Data/Dataset/Graph_data/abs_train'
    valid_path, valid_root = '../Data/Dataset/abs_test.csv', '../Data/Dataset/Graph_data/abs_test'

    SAVE_DIR = "../Data/model/abs/"
    LOG_FILE = f"../Data/Result/abs/abs_{args.solute_gnn}_{args.fusion_mode}.csv"

    trainer = Trainer(
        train_csv_path=train_path, valid_csv_path=valid_path,
        train_root_dir=train_root, valid_root_dir=valid_root,
        save_dir=SAVE_DIR,
        log_file=LOG_FILE
    )
    trainer.train(args.epochs)

