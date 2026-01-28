import csv
import os

import numpy as np
from matplotlib import pyplot as plt

from Graph_dataset import DatasetWithLabel
from GNN_module import GNNEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch_geometric.loader import DataLoader as DataLoaderGraph
from torch.utils.data import random_split
from torch import nn
import torch
from args import args
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(52)


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

        root = 'Data\Dataset/graph_data/7W14_demo112'
        self.dataset = DatasetWithLabel(root=root)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        self.train_loader = DataLoaderGraph(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoaderGraph(val_dataset, batch_size=args.batch_size, shuffle=False)

    def epoch_reg_train(self, epoch):
        self.model.train()
        train_loss_sum = 0.0
        loss_average = 0.0
        criterion = nn.MSELoss()

        all_label, all_prediction = [], []
        for step, batch in enumerate(self.train_loader):
            batch = batch.to(device)
            label = batch.label
            _, _, _, logit = self.model.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logit.squeeze(), label)
            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss_sum += float(loss.detach().cpu().item())
            loss_average = train_loss_sum / (step + 1)

            all_label.extend(label.detach().cpu())
            all_prediction.extend(logit.squeeze(dim=1).detach().cpu())
        r2 = r2_score(all_label, all_prediction)

        print(f'Epoch:{epoch}')
        print(f"===[Train Loss:  {loss_average:.3f}]===[Train R2:    {r2:.4f}]===")

        return loss_average, r2

    def epoch_reg_val(self):
        self.model.eval()
        val_loss_sum = 0.0
        loss_average = 0.0
        criterion = nn.MSELoss()

        all_label, all_prediction = [], []
        for step, batch in enumerate(self.val_loader):
            batch = batch.to(device)
            label = batch.label
            _, _, _, logit = self.model.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logit.squeeze(), label)
            val_loss_sum += float(loss.detach().cpu().item())
            loss_average = val_loss_sum / (step + 1)

            all_label.extend(label.detach().cpu().numpy())  # 转换为numpy数组
            all_prediction.extend(logit.squeeze().detach().cpu().numpy())

        r2 = r2_score(all_label, all_prediction)
        mse = mean_squared_error(all_label, all_prediction)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_label, all_prediction)

        mape = sum(
            abs((y_true - y_pred) / (y_true + 1e-8)) for y_true, y_pred in zip(all_label, all_prediction)) / len(
            all_label)

        print(f'===[Val Loss:    {loss_average:.3f}]===[Val R2:      {r2:.4f}]===')
        print(f'===[Val MSE:     {mse:.4f}]===[Val RMSE:    {rmse:.4f}]===')
        print(f'===[Val MAE:     {mae:.4f}]===[Val MAPE:    {mape:.4f}]===')

        return loss_average, r2, mse, rmse, mae, mape

    def save_model(self, epoch, val_r2):
        torch.save(self.model.state_dict(), rf'D:\PepBrick\GNN_Scoring\Data\Model/7W14_demo112/{args.gnn_type}_reg.pth')
        print('=' * 30)
        print(f"Save model at epoch: {epoch}, val R2: {val_r2:.3f}")
        print('=' * 30)

    def train(self, epochs):
        best_r2 = -1
        log_file = f'D:\PepBrick\GNN_Scoring\Data\Result/7W14_demo112/{args.gnn_type}_reg_log.csv'
        write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        for epoch in range(epochs):
            train_loss, train_r2 = self.epoch_reg_train(epoch)
            val_loss, val_r2, val_mse, val_rmse, val_mae, val_mape = self.epoch_reg_val()

            if val_r2 >= best_r2 and epoch > 10:
                self.save_model(epoch, val_r2)
                best_r2 = val_r2

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'epoch', 'train_loss', 'val_loss', 'train_r2', 'val_r2',
                        'val_mse', 'val_rmse', 'val_mae', 'val_mape'
                    ])
                    write_header = False
                writer.writerow([
                    epoch,
                    round(train_loss, 4),
                    round(val_loss, 4),
                    round(train_r2, 4),
                    round(val_r2, 4),
                    round(val_mse, 4),
                    round(val_rmse, 4),
                    round(val_mae, 4),
                    round(val_mape, 4)
                ])
























