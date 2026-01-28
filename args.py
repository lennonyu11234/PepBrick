import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--emb_dim', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=1e-2)
parser.add_argument('--max_len', type=int, default=100)

parser.add_argument('--gnn_num_layer', type=int, default=5)
parser.add_argument('--gnn_emb_dim', type=int, default=1024)

parser.add_argument('--JK', type=str, default='last',
                    help='optional: concat, last, max, sum')
parser.add_argument('--graph_pooling', type=str, default='mean',
                    help='optional: sum, mean, max, attention')
parser.add_argument('--gnn_type', type=str, default='GIN',
                    help='optional: GIN, GCN, GraphSAGE, GAT, GraphTransformer')
parser.add_argument('--mode', type=str, default='classification',
                    help='optional: classification, regression')

args = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


