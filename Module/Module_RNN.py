import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
import random
from Dataset_SMILES import idx_to_seqs, tokenizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class DecoderGRU(nn.Module):
    def __init__(self, voc_size):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, args.emb_dim)
        self.emb_proj = nn.Linear(args.emb_dim, args.emb_dim * 2)
        self.gru_1 = nn.GRUCell(args.emb_dim * 2, args.emb_dim * 2)
        self.gru_2 = nn.GRUCell(args.emb_dim * 2, args.emb_dim * 2)
        self.gru_3 = nn.GRUCell(args.emb_dim * 2, args.emb_dim * 2)
        self.ln1 = nn.LayerNorm(args.emb_dim * 2)
        self.ln2 = nn.LayerNorm(args.emb_dim * 2)
        self.ln3 = nn.LayerNorm(args.emb_dim * 2)
        self.fc1 = nn.Linear(args.emb_dim * 2, args.emb_dim * 4)
        self.fc2 = nn.Linear(args.emb_dim * 4, voc_size)
        self.dropout_emb = nn.Dropout(args.dropout if hasattr(args, 'dropout') else 0.1)
        self.dropout_out = nn.Dropout(args.dropout if hasattr(args, 'dropout') else 0.1)

    def forward(self, x, h):
        x = self.embedding(x)
        x = self.emb_proj(x)
        x = self.dropout_emb(x)
        h1 = self.gru_1(x, h[0])
        h1 = self.ln1(h1 + x)
        x = h1
        h2 = self.gru_2(x, h[1])
        h2 = self.ln2(h2 + x)
        x = h2
        h3 = self.gru_3(x, h[2])
        h3 = self.ln3(h3 + x)
        x = h3
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_out(x)
        x = self.fc2(x)
        h_out = torch.stack([h1, h2, h3], dim=0)
        return x.to(device), h_out.to(device)

    def init_h(self, batch_size):
        return torch.zeros(3, batch_size, args.emb_dim * 2, device=device)


class RNN(nn.Module):
    def __init__(self, voc):
        super().__init__()
        self.voc = voc
        voc_size = len(voc)
        self.rnn = DecoderGRU(voc_size)

        self.voc_idx2token = {v: k for k, v in self.voc.items()}

    def likelihood(self, target):
        device = target.device
        batch_size, seq_length = target.size()
        start_token = torch.full(
            (batch_size, 1),
            self.voc['<s>'],
            dtype=torch.long,
            device=device
        )

        x = torch.cat((start_token, target[:, :-1]), dim=1)
        h = self.rnn.init_h(batch_size).to(device)

        log_probs = torch.zeros(batch_size, device=device)

        for t in range(seq_length):
            logits, h = self.rnn(x[:, t], h)
            log_prob = F.log_softmax(logits, dim=-1)

            log_probs += log_prob.gather(
                1, target[:, t].unsqueeze(1)
            ).squeeze(1)

        return log_probs

    def sample(self, batch_size, max_len=100):
        start_token = torch.tensor(torch.zeros(batch_size).long())
        start_token[:] = self.voc['<s>']
        start_token.to(device)
        h = self.rnn.init_h(batch_size).to(device)
        x = start_token.to(device)

        sequences = []
        log_probs = torch.tensor(torch.zeros(batch_size)).to(device)
        finished = torch.zeros(batch_size).byte().to(device)
        entropy = torch.tensor(torch.zeros(batch_size)).to(device)

        for step in range(max_len):
            logits, h = self.rnn(x, h.to(device))
            prob = F.softmax(logits)
            log_prob = F.log_softmax(logits)
            x = torch.multinomial(prob, 1).view(-1)

            sequences.append(x.view(-1, 1))
            log_probs += NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = torch.tensor(x.data)
            EOS_sampled = (x == self.voc['</s>']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                break
        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy

    def sample_pattern(self, pattern="N[C@@H](C(=O)O)*", batch_size=1, max_len=100,
                       top_k=15,
                       temperature=1.0):
        pattern_tokens = tokenizer.tokenize(pattern)
        max_pattern_idx = len(pattern_tokens) - 1

        start_token = torch.full((batch_size,), fill_value=self.voc['<s>'], dtype=torch.long, device=device)
        input_tokens = start_token.clone()
        h = self.rnn.init_h(batch_size).to(device)

        sequences = [start_token.view(-1, 1)]

        log_probs = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        entropy = torch.zeros(batch_size, device=device)

        opening_parentheses = torch.zeros(batch_size, dtype=torch.int32, device=device)
        closing_parentheses = torch.zeros(batch_size, dtype=torch.int32, device=device)

        trackers = torch.zeros(batch_size, dtype=torch.long, device=device)
        generating_sidechain = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 记录每个样本在 * 处采样到的 token ids（不包含 EOS）
        sidechains = [[] for _ in range(batch_size)]

        vocab_size = len(self.voc)

        for step in range(max_len):
            current_pattern_indexes = [pattern_tokens[int(trackers[i].item())] for i in range(batch_size)]

            logits, h = self.rnn(input_tokens, h)
            prob = F.softmax(logits / temperature, dim=-1)
            log_prob = F.log_softmax(logits / temperature, dim=-1)
            if top_k > 0:
                topk_values, topk_indices = torch.topk(prob, top_k, dim=-1)
                topk_mask = torch.zeros_like(prob, device=device)
                topk_mask.scatter_(-1, topk_indices, 1)

                masked_prob = prob * topk_mask
                masked_prob = masked_prob / masked_prob.sum(dim=-1, keepdim=True)

                sampled = torch.multinomial(masked_prob, num_samples=1).view(-1)

            for i in range(batch_size):
                if finished[i]:
                    sampled[i] = self.voc['</s>']
                    continue

                cur_tok_str = current_pattern_indexes[i]

                if cur_tok_str == '*':
                    if not generating_sidechain[i]:
                        generating_sidechain[i] = True

                    if sampled[i].item() == self.voc['</s>']:
                        finished[i] = True
                        generating_sidechain[i] = False
                        continue

                    sidechains[i].append(sampled[i].item())

                    if self.voc.get('(') is not None and sampled[i].item() == self.voc['(']:
                        opening_parentheses[i] += 1
                    if self.voc.get(')') is not None and sampled[i].item() == self.voc[')']:
                        closing_parentheses[i] += 1

                    min_sidechain_len = 3
                    if (step > min_sidechain_len and
                            sampled[i].item() == self.voc.get(')') and
                            closing_parentheses[i] >= opening_parentheses[i]):
                        generating_sidechain[i] = False
                        trackers[i] = torch.clamp(trackers[i] + 1, max=max_pattern_idx)

                # 列表选择模式，例如 "[C,N,O]"
                elif isinstance(cur_tok_str, str) and cur_tok_str.startswith('[') and ',' in cur_tok_str:
                    choices_str = cur_tok_str[1:-1]
                    choices = [c.strip() for c in choices_str.split(',')]
                    mask = torch.zeros(vocab_size, device=device)
                    for ch in choices:
                        if ch in self.voc:
                            mask[self.voc[ch]] = 1.0

                    masked_probs = prob[i] * mask
                    if masked_probs.sum() > 0:
                        masked_probs /= masked_probs.sum()
                        sampled[i] = torch.multinomial(masked_probs.unsqueeze(0), 1).item()
                    else:
                        valid_idx = torch.nonzero(mask).view(-1)
                        if valid_idx.numel() > 0:
                            choice_idx = torch.randint(0, valid_idx.numel(), (1,)).item()
                            sampled[i] = valid_idx[choice_idx].item()

                    if trackers[i] < max_pattern_idx:
                        trackers[i] += 1

                # 精确指定 token 的情况
                else:
                    if cur_tok_str in self.voc:
                        forced_id = self.voc[cur_tok_str]
                        sampled[i] = forced_id
                        if trackers[i] < max_pattern_idx:
                            trackers[i] += 1

            sequences.append(sampled.view(-1, 1))

            log_probs += NLLLoss(log_prob, sampled)
            entropy -= torch.sum(log_prob * prob, dim=1)

            input_tokens = sampled.clone()
            EOS_sampled = (input_tokens == self.voc['</s>'])
            finished = finished | EOS_sampled

            if finished.all().item():
                break

        sequences = torch.cat(sequences, dim=1)
        # 构造最终拼接序列（pattern_before_star + sidechain + pattern_after_star）
        final_sequences = []
        # 尝试找到第一个 '*' 的位置（如果没有 '*' 则返回完整 pattern）
        try:
            star_idx = pattern_tokens.index('*')
        except ValueError:
            star_idx = None

        for i in range(batch_size):
            if star_idx is None:
                pattern_before_ids = [self.voc[t] for t in pattern_tokens if t in self.voc]
                pattern_after_ids = []
            else:
                pattern_before_ids = [self.voc[t] for t in pattern_tokens[:star_idx] if t in self.voc]
                pattern_after_ids = [self.voc[t] for t in pattern_tokens[star_idx + 1:] if t in self.voc]
            final_ids = pattern_before_ids + sidechains[i] + pattern_after_ids
            final_sequences.append(torch.tensor(final_ids, dtype=torch.long, device=device))

        return final_sequences, log_probs, entropy

    def sample_linker(self, batch_size, max_len=50, top_k=10, temperature=0.5):
        start_token = torch.full((batch_size,), fill_value=self.voc['<s>'], dtype=torch.long, device=device)
        input_tokens = start_token.clone()
        h = self.rnn.init_h(batch_size).to(device)
        sequences = [start_token.view(-1, 1)]
        log_probs = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        entropy = torch.zeros(batch_size, device=device)
        for step in range(max_len):
            logits, h = self.rnn(input_tokens, h)
            prob = F.softmax(logits / temperature, dim=-1)
            log_prob = F.log_softmax(logits / temperature, dim=-1)
            if top_k > 0:
                topk_values, topk_indices = torch.topk(prob, top_k, dim=-1)
                topk_mask = torch.zeros_like(prob, device=device)
                topk_mask.scatter_(-1, topk_indices, 1)
                masked_prob = prob * topk_mask
                masked_prob = masked_prob / masked_prob.sum(dim=-1, keepdim=True)  # 归一化

                sampled = torch.multinomial(masked_prob, num_samples=1).view(-1)
            else:
                sampled = torch.multinomial(prob, num_samples=1).view(-1)
            for i in range(batch_size):
                if finished[i]:
                    sampled[i] = self.voc['</s>']
            sequences.append(sampled.view(-1, 1))
            log_probs += F.nll_loss(log_prob, sampled, reduction='none')
            entropy -= torch.sum(log_prob * prob, dim=1)
            input_tokens = sampled.clone()
            EOS_sampled = (input_tokens == self.voc['</s>'])
            finished = finished | EOS_sampled
            if finished.all().item():
                break
        sequences = torch.cat(sequences, dim=1)
        return sequences.data, log_probs, entropy


def NLLLoss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    device = inputs.device
    target_expanded = torch.zeros_like(inputs, device=device)
    target_expanded.scatter_(1, targets.contiguous().view(-1, 1), 1.0)

    loss = (target_expanded * inputs).sum(dim=1)
    return loss





