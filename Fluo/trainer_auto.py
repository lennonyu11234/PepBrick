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
    def __init__(
        self,
        train_csv_path: str,
        valid_csv_path: str,
        train_root_dir: str,
        valid_root_dir: str,
        save_dir: str,
        log_file: str,
        task_name: str,
    ):
        self.task = task_name

        self.train_dataset = SoluteSolventDataset(root=train_root_dir, csv_path=train_csv_path)
        self.valid_dataset = SoluteSolventDataset(root=valid_root_dir, csv_path=valid_csv_path)

        self.train_loader = GeoDataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            follow_batch=["s_x"],
        )
        self.valid_loader = GeoDataLoader(
            self.valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            follow_batch=["s_x"],
        )

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
            fusion_dim=args.fusion_dim,
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
                y = data.label.view(-1).float()
                pred = self.model(data).view(-1).float()

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
        name = (
            f"best_{self.task}_"
            f"solute-{args.solute_gnn}_"
            f"solvent-{args.solvent_gnn}_"
            f"fusion-{args.fusion_mode}.pth"
        )
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, name))

    def train(self, epochs: int):
        best_r2 = -1e9
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        write_header = (not os.path.exists(self.log_file)) or (os.path.getsize(self.log_file) == 0)

        for epoch in tqdm(range(1, epochs + 1), total=epochs, desc="Training"):
            train_loss, train_r2 = self.train_epoch()
            val_mae, val_mse, val_rmse, val_r2 = self.val_epoch()

            print(
                f"\n[{self.task}] solute={args.solute_gnn} solvent={args.solvent_gnn} fusion={args.fusion_mode} | "
                f"epoch {epoch}/{epochs}\n"
                f"train loss:{train_loss:.4f} | train R2:{train_r2:.4f} | "
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
                w.writerow(
                    [
                        epoch,
                        round(train_loss, 6),
                        round(train_r2, 6),
                        round(val_mae, 6),
                        round(val_mse, 6),
                        round(val_rmse, 6),
                        round(val_r2, 6),
                    ]
                )


def build_paths_for_task(task: str):
    base_csv_dir = "../Data/Dataset"
    base_graph_dir = "../Data/Dataset/Graph_data"

    train_csv = os.path.join(base_csv_dir, f"{task}_train.csv")
    valid_csv = os.path.join(base_csv_dir, f"{task}_test.csv")

    train_root = os.path.join(base_graph_dir, f"{task}_train")
    valid_root = os.path.join(base_graph_dir, f"{task}_test")

    return train_csv, valid_csv, train_root, valid_root


if __name__ == "__main__":

    TASK_LIST = ['e', 'emi', 'abs', 'plqy']
    MODEL_LIST = [
        ("GIN", "GIN"),
        # ("GCN", "GCN"),
        # ("GAT", "GAT"),
        # ("GraphSAGE", "GraphSAGE"),
        # ("GraphTransformer", "GraphTransformer"),
    ]
    FUSION_LIST = [args.fusion_mode]

    base_save_dir = "../Data/model"
    base_result_dir = "../Data/Result"
    for task in TASK_LIST:
        train_csv, valid_csv, train_root, valid_root = build_paths_for_task(task)

        for fusion_mode in FUSION_LIST:
            args.fusion_mode = fusion_mode

            for solute_gnn, solvent_gnn in MODEL_LIST:
                args.solute_gnn = solute_gnn
                args.solvent_gnn = solvent_gnn

                SAVE_DIR = os.path.join(
                    base_save_dir,
                    task,
                    f"solute-{solute_gnn}_solvent-{solvent_gnn}",
                    fusion_mode,
                )
                LOG_FILE = os.path.join(
                    base_result_dir,
                    task,
                    f"{task}_solute-{solute_gnn}_solvent-{solvent_gnn}_fusion-{fusion_mode}.csv",
                )

                os.makedirs(SAVE_DIR, exist_ok=True)
                os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

                print("\n" + "=" * 90)
                print(f"RUN | task={task} | solute_gnn={solute_gnn} | solvent_gnn={solvent_gnn} | fusion={fusion_mode}")
                print(f"train_csv:  {train_csv}")
                print(f"valid_csv:  {valid_csv}")
                print(f"train_root: {train_root}")
                print(f"valid_root: {valid_root}")
                print(f"save_dir:   {SAVE_DIR}")
                print(f"log_file:   {LOG_FILE}")
                print("=" * 90 + "\n")

                trainer = Trainer(
                    train_csv_path=train_csv,
                    valid_csv_path=valid_csv,
                    train_root_dir=train_root,
                    valid_root_dir=valid_root,
                    save_dir=SAVE_DIR,
                    log_file=LOG_FILE,
                    task_name=task,
                )
                trainer.train(args.epochs)
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
