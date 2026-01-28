import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 120
num_chirality_tag = 4
num_bond_type = 7
num_bond_direction = 4

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


class GraphSageLayer(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super().__init__()
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GINLayer(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__()
        self.mlp = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_emb, aggr=self.aggr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNLayer(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__()
        self.mlp = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        row, _ = edge_index
        num_nodes = x.size(0)
        deg = torch.bincount(row, minlength=num_nodes).to(dtype=x.dtype, device=x.device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[row]

        x = self.mlp(x)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_emb, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATLayer(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(node_dim=0)
        self.att = nn.Linear(2 * emb_dim, 1)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.att.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_i, x_j, edge_attr):
        x_j = x_j + edge_attr
        att_in = torch.cat([x_i, x_j], dim=-1)
        a = self.att(att_in).sigmoid()
        return x_j * a

    def update(self, aggr_out):
        return F.relu(aggr_out)


class GraphTransformerLayer(MessagePassing):
    def __init__(self, emb_dim, num_heads=4):
        super().__init__(node_dim=0)
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.qkv = nn.Linear(emb_dim, 3 * emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)

        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        res = x
        x = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_emb)
        x = self.ln1(x + res)

        res = x
        x = self.ffn(x)
        x = self.ln2(x + res)
        return x

    def message(self, x_i, x_j, edge_attr):
        x_j = x_j + edge_attr
        E = x_j.size(0)
        qkv = self.qkv(x_j).view(E, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scores = (q * k).sum(-1) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=0)
        out = (v * scores.unsqueeze(-1)).reshape(E, self.emb_dim)
        return self.out(out)


# =========================
# GNN Encoder
# =========================
class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.1, gnn_type="GIN"):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.x_emb1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_emb2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_emb1.weight.data)
        nn.init.xavier_uniform_(self.x_emb2.weight.data)

        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "GraphSAGE":
                self.layers.append(GraphSageLayer(emb_dim))
            elif gnn_type == "GIN":
                self.layers.append(GINLayer(emb_dim))
            elif gnn_type == "GCN":
                self.layers.append(GCNLayer(emb_dim))
            elif gnn_type == "GAT":
                self.layers.append(GATLayer(emb_dim))
            elif gnn_type == "GraphTransformer":
                self.layers.append(GraphTransformerLayer(emb_dim))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.bns = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

    def forward(self, x, edge_index, edge_attr):
        h = self.x_emb1(x[:, 0]) + self.x_emb2(x[:, 1])
        h_list = [h]

        for layer in range(self.num_layer):
            h = self.layers[layer](h_list[layer], edge_index, edge_attr)
            h = self.bns[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            return h_list[-1]
        if self.JK == "sum":
            return torch.stack(h_list, dim=0).sum(dim=0)
        if self.JK == "max":
            return torch.stack(h_list, dim=0).max(dim=0)[0]
        if self.JK == "concat":
            return torch.cat(h_list, dim=1)
        raise ValueError(f"Unknown JK: {self.JK}")


class GNNEncoder(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.1, pooling="mean", gnn_type="GIN"):
        super().__init__()
        self.gnn = GNN(num_layer=num_layer, emb_dim=emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        if JK == "concat":
            raise ValueError("This runnable baseline expects JK != 'concat' for simplicity.")

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_h = self.gnn(x, edge_index, edge_attr)
        g = self.pool(node_h, batch)
        g = self.proj(g)
        return g


class Fusion(nn.Module):
    def __init__(self, g_dim, s_dim, out_dim, mode="concat", drop=0.1):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(drop)

        if mode == "concat":
            self.proj = nn.Linear(g_dim + s_dim, out_dim)

        elif mode == "gated":
            self.gate = nn.Sequential(nn.Linear(s_dim, g_dim), nn.Sigmoid())
            self.proj = nn.Linear(g_dim + s_dim, out_dim)

        elif mode == "cross_attn":
            self.Wq = nn.Linear(s_dim, out_dim)
            self.Wk = nn.Linear(g_dim, out_dim)
            self.Wv = nn.Linear(g_dim, out_dim)
            self.proj = nn.Linear(out_dim + s_dim, out_dim)

        elif mode == "gru":
            self.gru = nn.GRU(input_size=out_dim, hidden_size=out_dim, batch_first=True)
            self.in_proj_g = nn.Linear(g_dim, out_dim)
            self.in_proj_s = nn.Linear(s_dim, out_dim)
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, g, s):
        if self.mode == "concat":
            z = torch.cat([g, s], dim=-1)
            return self.proj(self.drop(z))

        if self.mode == "gated":
            gate = self.gate(s)
            g2 = g * gate
            z = torch.cat([g2, s], dim=-1)
            return self.proj(self.drop(z))

        if self.mode == "cross_attn":
            q = self.Wq(s).unsqueeze(1)   # [B,1,d]
            k = self.Wk(g).unsqueeze(1)
            v = self.Wv(g).unsqueeze(1)
            attn = torch.softmax((q * k).sum(-1, keepdim=True) / math.sqrt(q.size(-1)), dim=1)  # [B,1,1]
            a = (attn * v).squeeze(1)  # [B,d]
            z = torch.cat([a, s], dim=-1)
            return self.proj(self.drop(z))

        if self.mode == "gru":
            g0 = self.in_proj_g(g)
            s0 = self.in_proj_s(s)
            seq = torch.stack([g0, s0], dim=1)  # [B,2,d]
            _, h = self.gru(seq)
            return h.squeeze(0)


class MorganEncoder(nn.Module):
    def __init__(self, n_bits=2048, out_dim=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, out_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, fp):
        return self.net(fp)


class GraphSolventScaffoldRegressor(nn.Module):
    def __init__(self,
                 emb_dim=256,
                 num_layer=5,
                 drop_ratio=0.1,
                 JK="last",
                 pooling="mean",
                 solute_gnn="GraphSAGE",
                 solvent_gnn="GraphSAGE",
                 use_scaffold=True,
                 scaffold_vocab=2048,
                 scaffold_emb=128,
                 fusion_mode="gru",
                 fusion_dim=256,
                 morgan_bits=2048,
                 morgan_dim=None):
        super().__init__()

        self.use_scaffold = use_scaffold
        if morgan_dim is None:
            morgan_dim = emb_dim

        self.solute_enc = GNNEncoder(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, pooling=pooling, gnn_type=solute_gnn)
        self.solvent_enc = GNNEncoder(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, pooling=pooling, gnn_type=solvent_gnn)

        self.g_morgan_enc = MorganEncoder(n_bits=morgan_bits, out_dim=morgan_dim, drop=drop_ratio)
        self.s_morgan_enc = MorganEncoder(n_bits=morgan_bits, out_dim=morgan_dim, drop=drop_ratio)

        if use_scaffold:
            self.scaffold_embedding = nn.Embedding(scaffold_vocab, scaffold_emb)
            nn.init.xavier_uniform_(self.scaffold_embedding.weight.data)
        else:
            self.scaffold_embedding = None

        solute_in = emb_dim + morgan_dim + (scaffold_emb if use_scaffold else 0)
        self.solute_feat_proj = nn.Sequential(
            nn.Linear(solute_in, emb_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim, emb_dim),
        )

        solvent_in = emb_dim + morgan_dim
        self.solvent_feat_proj = nn.Sequential(
            nn.Linear(solvent_in, emb_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim, emb_dim),
        )

        self.fusion = Fusion(
            g_dim=emb_dim,
            s_dim=emb_dim,
            out_dim=fusion_dim,
            mode=fusion_mode,
            drop=drop_ratio
        )

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, data: Data):
        g_graph = self.solute_enc(data.x, data.edge_index, data.edge_attr, data.batch)  # [B, emb_dim]
        s_graph = self.solvent_enc(data.s_x, data.s_edge_index, data.s_edge_attr, data.s_x_batch)  # [B, emb_dim]
        g_fp = data.g_morgan.float()
        s_fp = data.s_morgan.float()
        g_morgan = self.g_morgan_enc(g_fp)
        s_morgan = self.s_morgan_enc(s_fp)

        if self.use_scaffold:
            sc = self.scaffold_embedding(data.scaffold_id)
            g_feat = torch.cat([g_graph, g_morgan, sc], dim=-1)
        else:
            g_feat = torch.cat([g_graph, g_morgan], dim=-1)

        s_feat = torch.cat([s_graph, s_morgan], dim=-1)

        g = self.solute_feat_proj(g_feat)
        s = self.solvent_feat_proj(s_feat)
        fused = self.fusion(g, s)
        pred = self.head(fused).squeeze(-1)
        return pred
