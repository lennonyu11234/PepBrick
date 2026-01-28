import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
import torch.nn.functional as F
from GNN_Scoring.Graph_dataset import count_lines
from args import args
num_atom_type = 120
num_chirality_tag = 4
num_bond_type = 7
num_bond_direction = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = count_lines(args.vocab) + 20


class AddNorm(nn.Module):
    def __init__(self, normalize_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(normalize_shape)

    def forward(self, X, Y):
        Y = self.dropout(Y) + X
        return self.normalize(Y)


class FFN(nn.Module):
    def __init__(self, ffn_inputs, ffn_hiddens=1024, ffn_outputs=512):
        super().__init__()
        self.dense1 = nn.Linear(ffn_inputs, ffn_hiddens)
        self.dense2 = nn.Linear(ffn_hiddens, ffn_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.dense2(self.relu(self.dense1(X)))
        return X.to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout=0.0, desc='enc'):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.normalize = AddNorm(num_hiddens, dropout)
        self.desc = desc

        # define q,k,v linear layer
        self.Wq = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wk = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wv = nn.Linear(self.num_hiddens, self.num_hiddens)

        self.relu = nn.ReLU()
        self.Q = nn.Sequential(self.Wq, self.relu)
        self.K = nn.Sequential(self.Wk, self.relu)
        self.V = nn.Sequential(self.Wv, self.relu)

    def forward(self, queries, keys, values):
        # get matrices of q, k, v
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        q_split = q.unsqueeze(1).chunk(self.num_heads, dim=-1)
        k_split = k.unsqueeze(1).chunk(self.num_heads, dim=-1)
        v_split = v.unsqueeze(1).chunk(self.num_heads, dim=-1)

        q_stack = torch.stack(q_split, dim=1)
        k_stack = torch.stack(k_split, dim=1)
        v_stack = torch.stack(v_split, dim=1)

        score = torch.matmul(q_stack, k_stack.permute(0, 1, 3, 2))
        score = score / (k_stack.size()[-1] ** 0.5)

        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v_stack)
        a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1)))
        a += queries

        return a


class BridgeTowerBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout_rate):
        super().__init__()
        self.self_attetion_1 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)
        self.self_attetion_2 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)

        self.cross_attetion_1 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)
        self.cross_attetion_2 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)

        self.bridge_layer_1 = nn.Linear(num_hiddens, num_hiddens)
        self.bridge_layer_2 = nn.Linear(num_hiddens, num_hiddens)

        self.ffn_1 = FFN(num_hiddens)
        self.ffn_2 = FFN(num_hiddens)

        self.AddNorm1 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm2 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm3 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm4 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm5 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm6 = AddNorm(num_hiddens, dropout=dropout_rate)

    def forward(self, modality_1, modality_2):
        output_1 = self.bridge_layer_1(modality_1)
        output_2 = self.bridge_layer_2(modality_2)

        output_attn_1 = self.self_attetion_1(output_1, output_1, output_1)
        output_attn_2 = self.self_attetion_2(output_2, output_2, output_2)

        modality_1 = self.AddNorm1(modality_1, output_attn_1)
        modality_2 = self.AddNorm2(modality_2, output_attn_2)

        output_attn_1 = self.cross_attetion_1(modality_1, modality_2, modality_2)
        output_attn_2 = self.cross_attetion_2(modality_2, modality_1, modality_1)

        modality_1 = self.AddNorm3(modality_1, output_attn_1)
        modality_2 = self.AddNorm4(modality_2, output_attn_2)

        output1 = self.ffn_1(modality_1)
        output2 = self.ffn_2(modality_2)

        output1 = self.AddNorm5(modality_1, output1)
        output2 = self.AddNorm6(modality_2, output2)

        return output1, output2


class GraphSageLayer(MessagePassing):
    def __init__(self, emb_dim, aggr='mean'):
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embedding = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index=edge_index, x=x,
                              edge_attr=edge_embedding, aggr=self.aggr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GINLayer(MessagePassing):
    def __init__(self, emb_dim, aggr='add'):
        super().__init__()
        self.mlp = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index = edge_index.to(x.device).to(torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(edge_attr.device).to(edge_attr.dtype)
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4

        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embedding = self.edge_embedding1(edge_attr[:, 0]) + \
                         self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index=edge_index.to(torch.int64), x=x,
                              edge_attr=edge_embedding, aggr=self.aggr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNLayer(MessagePassing):
    def __init__(self, emb_dim, aggr='add'):
        super().__init__()
        self.mlp = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # Calculate normalization directly without using scatter_add
        row, _ = edge_index
        num_nodes = x.size(0)
        dtype = x.dtype
        deg = torch.bincount(row, minlength=num_nodes).to(dtype=dtype, device=x.device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[row]

        x = self.mlp(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm,
                              aggr=self.aggr)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATLayer(MessagePassing):
    def __init__(self, emb_dim, aggr='add'):
        super().__init__(node_dim=0)
        self.emb_dim = emb_dim
        self.att = nn.Linear(2 * emb_dim, 1)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.att.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index = edge_index.to(x.device).to(torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(edge_attr.device).to(edge_attr.dtype)
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embedding = self.edge_embedding1(edge_attr[:, 0]) + \
                         self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index=edge_index.to(torch.int64),
                              x=x,
                              edge_attr=edge_embedding)

    def message(self, x_i, x_j, edge_attr):
        x_j = x_j + edge_attr
        att_input = torch.cat([x_i, x_j], dim=-1)
        att_score = self.att(att_input).sigmoid()
        return x_j * att_score

    def update(self, aggr_out):
        return F.relu(aggr_out)


class GraphTransformerLayer(MessagePassing):
    def __init__(self, emb_dim, num_heads=4, aggr='add'):
        super().__init__(node_dim=0)
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index = edge_index.to(x.device).to(torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embedding = self.edge_embedding1(edge_attr[:, 0].long()) + \
                         self.edge_embedding2(edge_attr[:, 1].long())
        residual = x
        x = self.propagate(edge_index=edge_index.to(torch.int64),
                           x=x,
                           edge_attr=edge_embedding)
        x = self.ln1(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.ln2(x + residual)
        return x

    def message(self, x_i, x_j, edge_attr):
        x_j = x_j + edge_attr
        batch_size = x_j.size(0)
        qkv = self.qkv_proj(x_j).view(batch_size, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        attn_scores = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=0)
        attn_output = (v * attn_scores.unsqueeze(-1)).view(batch_size, self.emb_dim)
        return self.out_proj(attn_output)

    def update(self, aggr_out):
        return aggr_out


class GNN(nn.Module):
    def __init__(self,
                 num_layer,
                 emb_dim,
                 JK='last',
                 drop_ratio=0.1,
                 gnn_type='gin'):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnn = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == 'GraphSAGE':
                self.gnn.append(GraphSageLayer(emb_dim))
            elif gnn_type == 'GIN':
                self.gnn.append(GINLayer(emb_dim))
            elif gnn_type == 'GCN':
                self.gnn.append(GCNLayer(emb_dim))
            elif gnn_type == 'GAT':
                self.gnn.append(GATLayer(emb_dim))
            elif gnn_type == 'GraphTransformer':
                self.gnn.append(GraphTransformerLayer(emb_dim))

        self.batch_norm = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norm.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        embed1 = self.x_embedding1(x[:, 0])
        embed2 = self.x_embedding2(x[:, 1])
        x = embed1 + embed2
        h_list = [x]

        for layer in range(self.num_layer):
            h = self.gnn[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norm[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            cat_h_list = torch.cat(h_list, dim=0)
            node_representation = torch.max(cat_h_list, dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            cat_h_list = torch.cat(h_list, dim=0)
            node_representation = torch.sum(cat_h_list, dim=0)

        return node_representation


class GNNEncoder(nn.Module):
    def __init__(self,
                 num_layer,
                 emb_dim,
                 JK='last',
                 drop_ratio=0,
                 graph_pooling='mean',
                 gnn_type='gcn',
                 func='classification'):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.func = func
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))

        self.projection_head = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(emb_dim, emb_dim))

        if self.func == 'classification':
            self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim//2),
                                     nn.ReLU(),
                                     nn.Linear(emb_dim//2, 2))
        elif self.func == 'regression':
            self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                                     nn.ReLU(),
                                     nn.Linear(emb_dim // 2, 1))

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch.long())
        graph_representation = self.projection_head(graph_representation)

        graph_logit = self.mlp(graph_representation)
        node_logit = self.mlp(node_representation)

        return node_representation, graph_representation, node_logit, graph_logit




