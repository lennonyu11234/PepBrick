import csv
import os
from Graph_dataset import DatasetWithLabel
from GNN_module import GNNEncoder
from sklearn.metrics import precision_score, f1_score, recall_score, matthews_corrcoef, accuracy_score
from torch_geometric.loader import DataLoader as DataLoaderGraph
from torch.utils.data import random_split
from torch import nn
import torch
from args import args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNTrainer:
    def __init__(self):
        self.model = GNNEncoder(num_layer=args.gnn_num_layer,
                                emb_dim=args.gnn_emb_dim,
                                JK=args.JK,
                                drop_ratio=args.dropout,
                                graph_pooling=args.graph_pooling,
                                gnn_type=args.gnn_type,
                                func=args.mode).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scaler = torch.cuda.amp.GradScaler()

        train_root = 'Data\Dataset/graph_data/Neuro_train'
        train_dataset = DatasetWithLabel(root=train_root)
        val_root = 'Data\Dataset/graph_data/Neuro_test'
        val_dataset = DatasetWithLabel(root=val_root)
        self.train_loader = DataLoaderGraph(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoaderGraph(val_dataset, batch_size=args.batch_size, shuffle=False)

    def epoch_train_cls(self, epoch):
        self.model.train()
        train_loss_sum = 0.0
        all_label, all_prediction = [], []

        for step, batch in enumerate(self.train_loader):
            batch = batch.to(device)
            label = torch.where(batch.label >= 0.5, torch.tensor(1), torch.tensor(0))
            _, _, _, logit = self.model(batch.x, batch.edge_index,
                                        batch.edge_attr, batch.batch)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logit, label)

            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss_sum += loss.item()
            loss_avg = train_loss_sum / (step + 1)

            all_label.extend(label.cpu())
            all_prediction.extend(logit.argmax(dim=1).cpu())

        acc = accuracy_score(all_label, all_prediction)
        print(f"Epoch {epoch}:")
        print(f"Train Loss {loss_avg:.3f} | Acc {acc:.3f}")
        return train_loss_sum

    def epoch_val_cls(self):
        self.model.eval()
        val_loss_sum = 0.0
        all_label, all_prediction = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                label = torch.where(batch.label >= 0.5, torch.tensor(1), torch.tensor(0))
                _, _, _, logit = self.model(batch.x, batch.edge_index,
                                            batch.edge_attr, batch.batch)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logit, label)
                val_loss_sum += loss.item()

                all_label.extend(label.cpu())
                all_prediction.extend(logit.argmax(dim=1).cpu())

        acc = accuracy_score(all_label, all_prediction)
        precision = precision_score(all_label, all_prediction)
        f1 = f1_score(all_label, all_prediction)
        mcc = matthews_corrcoef(all_label, all_prediction)
        recall = recall_score(all_label, all_prediction)
        print(
            f"Val Loss   {val_loss_sum / len(self.val_loader):.3f} | Acc {acc:.3f} | Pre {precision:.3f} | Rec {recall:.3f} | F1 {f1:.3f} | MCC {mcc:.3f}")
        return val_loss_sum, acc, precision, recall, f1, mcc

    def save_model(self, epoch, val_r2):
        torch.save(self.model.state_dict(), f'Data\Model/Neuro/{args.gnn_type}_cls.pth')
        print('=' * 30)
        print(f"Save model at epoch: {epoch}, val R2: {val_r2:.3f}")
        print('=' * 30)

    def train(self, epochs):
        best_acc = 0
        log_file = f'Data\Result/Neuro/{args.gnn_type}_cls_log.csv'
        write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        for epoch in range(epochs):
            train_loss = self.epoch_train_cls(epoch)
            val_loss, val_acc, val_pre, val_rec, val_f1, val_mcc = self.epoch_val_cls()

            if val_acc >= best_acc and epoch > 10:
                self.save_model(epoch, val_acc)
                best_acc = val_acc

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'epoch', 'train_loss', 'val_loss', 'val_ACC',
                        'val_Pre', 'val_Rec', 'val_F1', 'val_MCC'
                    ])
                    write_header = False
                writer.writerow([
                    epoch,
                    round(train_loss, 4),
                    round(val_loss, 4),
                    round(val_acc, 4),
                    round(val_pre, 4),
                    round(val_rec, 4),
                    round(val_f1, 4),
                    round(val_mcc, 4)
                ])

























