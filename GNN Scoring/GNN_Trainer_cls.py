import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from torch import nn
from torch_geometric.loader import DataLoader as DataLoaderGraph

from Graph_dataset import DatasetWithLabel
from GNN_module import GNNEncoder
from args import args


TRAIN_ROOT = r"Data/Dataset/graph_data/train"
VAL_ROOT = r"Data/Dataset/graph_data/valid"
TEST_ROOT = r"Data/Dataset/graph_data/test"
OUTPUT_DIR = rf"Data/Model/{args.gnn_type}_cls"
LABEL_THRESHOLD = 0.5
POSITIVE_IF_HIGHER = True
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path, content):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            content, file, indent=2,
            ensure_ascii=False, allow_nan=False
        )


def classification_metrics(labels, predictions):
    return {
        "Acc": float(accuracy_score(labels, predictions)),
        "Precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(labels, predictions)),
    }


class GNNTrainer:
    def __init__(
        self,
        train_root=TRAIN_ROOT,
        val_root=VAL_ROOT,
        test_root=TEST_ROOT,
        output_dir=OUTPUT_DIR,
        threshold=LABEL_THRESHOLD,
        positive_if_higher=POSITIVE_IF_HIGHER,
        seed=SEED,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.threshold = threshold
        self.positive_if_higher = positive_if_higher
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
            if len(dataset) == 0:
                raise ValueError(f"{name} must not be empty.")
        for name, dataset in (
            ("train", self.train_dataset),
            ("validation", self.val_dataset),
        ):
            values = torch.cat([
                graph.label.reshape(-1) for graph in dataset
            ])
            if self.binary_labels(values).unique().numel() != 2:
                raise ValueError(
                    f"{name} must contain both classes after thresholding."
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
            "func": "classification",
        }

        self.model = GNNEncoder(**architecture).to(device)

        weight_decay = getattr(args, "weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            "task": "classification",
            "architecture": architecture,
            "optimization": {
                "optimizer": "Adam",
                "lr": args.lr,
                "weight_decay": weight_decay,
                "batch_size": args.batch_size,
                "loss": "CrossEntropyLoss",
                "gradient_clip_norm": 1.0,
                "mixed_precision": False,
            },
            "label_threshold": threshold,
            "positive_if_higher": positive_if_higher,
            "positive_class_definition": (
                f"Label >= {threshold}"
                if positive_if_higher
                else f"Label <= {threshold}"
            ),
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
            "selection_metric": "maximum validation accuracy",
            "torch_version": torch.__version__,
        }

    def binary_labels(self, values):
        values = values.reshape(-1)
        if self.positive_if_higher:
            return (values >= self.threshold).long()
        return (values <= self.threshold).long()

    def epoch_train_cls(self):
        self.model.train()
        loss_sum = 0.0
        all_labels, all_predictions = [], []

        for batch in self.train_loader:
            batch = batch.to(device)
            labels = self.binary_labels(batch.label)

            self.optimizer.zero_grad()

            logits = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )[-1]
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            loss_sum += loss.item() * labels.numel()
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(
                logits.detach().argmax(dim=1).cpu().tolist()
            )

        train_loss = loss_sum / len(self.train_dataset)
        train_acc = float(
            accuracy_score(all_labels, all_predictions)
        )
        return train_loss, train_acc

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        loss_sum = 0.0
        all_labels, all_predictions, all_probabilities = [], [], []
        for batch in loader:
            batch = batch.to(device)
            labels = self.binary_labels(batch.label)

            logits = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )[-1]
            loss_sum += (
                self.criterion(logits, labels).item()
                * labels.numel()
            )
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )
            all_probabilities.extend(
                logits.softmax(dim=1)[:, 1].cpu().tolist()
            )
        metrics = classification_metrics(
            all_labels, all_predictions
        )
        metrics["Loss"] = loss_sum / len(all_labels)
        return metrics, all_labels, all_predictions, all_probabilities

    def train(self, epochs):
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        self.config["optimization"]["epochs"] = epochs
        save_json(self.output_dir / "config.json", self.config)
        best_acc = -float("inf")
        fields = [
            "epoch",
            "train_loss",
            "train_Acc",
            "val_loss",
            "val_Acc",
            "val_Precision",
            "val_Recall",
            "val_F1",
            "val_MCC",
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
                train_loss, train_acc = self.epoch_train_cls()
                val_metrics, _, _, _ = self.evaluate(self.val_loader)

                val_record = {
                    "val_loss": val_metrics["Loss"],
                    **{
                        f"val_{name}": value
                        for name, value in val_metrics.items()
                        if name != "Loss"
                    },
                }
                if val_metrics["Acc"] > best_acc:
                    best_acc = val_metrics["Acc"]
                    torch.save(
                        self.model.state_dict(),
                        self.output_dir / "best.pth",
                    )
                    save_json(
                        self.output_dir / "best_validation.json",
                        {
                            "epoch": epoch,
                            **val_record,
                        },
                    )
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_Acc": train_acc,
                    **val_record,
                })
                file.flush()
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Acc: {val_metrics['Acc']:.4f} | "
                    f"Val F1: {val_metrics['F1']:.4f} | "
                    f"Val MCC: {val_metrics['MCC']:.4f}",
                    flush=True,
                )
        self.model.load_state_dict(
            torch.load(
                self.output_dir / "best.pth",
                map_location=device,
                weights_only=True,
            )
        )
        test_metrics, labels, predictions, probabilities = (
            self.evaluate(self.test_loader)
        )
        save_json(
            self.output_dir / "test_metrics.json",
            test_metrics,
        )
        pd.DataFrame({
            "Label": labels,
            "Prediction": predictions,
            "Positive_probability": probabilities,
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
        threshold=LABEL_THRESHOLD,
        positive_if_higher=POSITIVE_IF_HIGHER,
        seed=SEED,
    )
    trainer.train(epochs=args.epochs)