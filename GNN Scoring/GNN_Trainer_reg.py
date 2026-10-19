import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch_geometric.loader import DataLoader as DataLoaderGraph

from Graph_dataset import DatasetWithLabel
from GNN_module import GNNEncoder
from args import args


TRAIN_ROOT = r"Data/Dataset/graph_data/train"
VAL_ROOT = r"Data/Dataset/graph_data/valid"
TEST_ROOT = r"Data/Dataset/graph_data/test"
OUTPUT_DIR = rf"Data/Model/{args.gnn_type}_reg"
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_json(path, content):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            content, file, indent=2,
            ensure_ascii=False, allow_nan=False
        )


def regression_metrics(labels, predictions):
    mse = mean_squared_error(labels, predictions)
    return {
        "R2": float(r2_score(labels, predictions)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(labels, predictions)),
    }


class GNNTrainer:
    def __init__(
        self,
        train_root=TRAIN_ROOT,
        val_root=VAL_ROOT,
        test_root=TEST_ROOT,
        output_dir=OUTPUT_DIR,
        seed=SEED,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        roots = [
            Path(path).resolve()
            for path in (train_root, val_root, test_root)
        ]
        for root in roots:
            path = root / "processed" / "processed_data.dataset"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Prepare the graph dataset first: {path}"
                )
        self.train_dataset, self.val_dataset, self.test_dataset = [
            DatasetWithLabel(root=str(root)) for root in roots
        ]
        datasets = (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        )
        for name, dataset in zip(("train", "validation", "test"), datasets):
            if len(dataset) < 2:
                raise ValueError(
                    f"{name} must contain at least two samples for R2."
                )
        self.train_loader = DataLoaderGraph(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoaderGraph(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        self.test_loader = DataLoaderGraph(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        architecture = {
            "num_layer": args.gnn_num_layer,
            "emb_dim": args.gnn_emb_dim,
            "JK": args.JK,
            "drop_ratio": args.dropout,
            "graph_pooling": args.graph_pooling,
            "gnn_type": args.gnn_type,
            "func": "regression",
        }
        self.model = GNNEncoder(**architecture).to(device)
        weight_decay = getattr(args, "weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=weight_decay,
        )
        self.criterion = nn.MSELoss()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = {
            "task": "regression",
            "architecture": architecture,
            "optimization": {
                "optimizer": "Adam",
                "lr": args.lr,
                "weight_decay": weight_decay,
                "batch_size": args.batch_size,
                "loss": "MSELoss",
                "gradient_clip_norm": 1.0,
                "mixed_precision": False,
            },
            "label_threshold": None,
            "label_transform": "Stored graph labels are used unchanged.",
            "partitions": {
                name: {
                    "root": str(root),
                    "n_records": len(dataset),
                }
                for name, root, dataset in zip(
                    ("train", "validation", "test"),
                    roots,
                    datasets,
                )
            },
            "seed": seed,
            "device": str(device),
            "selection_metric": "maximum validation R2",
            "torch_version": torch.__version__,
        }

    def epoch_reg_train(self):
        self.model.train()
        loss_sum = 0.0
        all_labels, all_predictions = [], []
        for batch in self.train_loader:
            batch = batch.to(device)
            labels = batch.label.float().reshape(-1)
            self.optimizer.zero_grad()
            predictions = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )[-1].reshape(-1)
            loss = self.criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            loss_sum += loss.item() * labels.numel()
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
        train_loss = loss_sum / len(self.train_dataset)
        train_r2 = float(r2_score(all_labels, all_predictions))
        return train_loss, train_r2

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_labels, all_predictions = [], []
        for batch in loader:
            batch = batch.to(device)
            labels = batch.label.float().reshape(-1)
            predictions = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )[-1].reshape(-1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
        metrics = regression_metrics(all_labels, all_predictions)
        return metrics, all_labels, all_predictions
    def train(self, epochs):
        self.config["optimization"]["epochs"] = epochs
        save_json(self.output_dir / "config.json", self.config)
        best_r2 = -float("inf")
        fields = [
            "epoch",
            "train_loss",
            "train_R2",
            "val_loss",
            "val_R2",
            "val_MSE",
            "val_RMSE",
            "val_MAE",
        ]
        with open(
            self.output_dir / "training_log.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                train_loss, train_r2 = self.epoch_reg_train()
                val_metrics, _, _ = self.evaluate(self.val_loader)
                if val_metrics["R2"] > best_r2:
                    best_r2 = val_metrics["R2"]
                    torch.save(
                        self.model.state_dict(),
                        self.output_dir / "best.pth",
                    )
                    save_json(
                        self.output_dir / "best_validation.json",
                        {
                            "epoch": epoch,
                            "val_loss": val_metrics["MSE"],
                            **{
                                f"val_{name}": value
                                for name, value in val_metrics.items()
                            },
                        },
                    )
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_R2": train_r2,
                    "val_loss": val_metrics["MSE"],
                    **{
                        f"val_{name}": value
                        for name, value in val_metrics.items()
                    },
                })
                file.flush()
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val R2: {val_metrics['R2']:.4f} | "
                    f"Val RMSE: {val_metrics['RMSE']:.4f} | "
                    f"Val MAE: {val_metrics['MAE']:.4f}",
                    flush=True,
                )
        self.model.load_state_dict(
            torch.load(
                self.output_dir / "best.pth",
                map_location=device,
                weights_only=True,
            )
        )
        test_metrics, labels, predictions = self.evaluate(self.test_loader)
        save_json(
            self.output_dir / "test_metrics.json",
            test_metrics,
        )
        pd.DataFrame({
            "Label": labels,
            "Prediction": predictions,
        }).to_csv(
            self.output_dir / "test_predictions.csv",
            index=False,
        )
        print(f"Test metrics: {test_metrics}")
        print(f"Results saved to: {self.output_dir.resolve()}")
        return test_metrics


if __name__ == "__main__":
    trainer = GNNTrainer(
        train_root=TRAIN_ROOT,
        val_root=VAL_ROOT,
        test_root=TEST_ROOT,
        output_dir=OUTPUT_DIR,
        seed=SEED,
    )
    trainer.train(epochs=args.epochs)