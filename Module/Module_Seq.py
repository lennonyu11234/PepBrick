import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
import copy
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
from Utils.Merge_AA_SMILES import process_smiles_chain
from Module.Module_RNN import RNN
from Dataset_SMILES import tokenizer
from Module.Cyclic_utils import add_explicit_hydrogens
from rdkit.Chem import QED
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Natural_AA = {
    'A': 'N[C@@H](C(=O)O)C',
    'R': 'N[C@@H](C(=O)O)(CCC/N=C(N[H])/N)',
    'N': 'N[C@@H](C(=O)O)CC(=O)(N)',
    'D': 'N[C@@H](C(=O)O)CC(=O)(O)',
    'C': 'N[C@@H](C(=O)O)CS',
    'Q': 'N[C@@H](C(=O)O)CCC(=O)(N)',
    'E': 'N[C@@H](C(=O)O)CCC(=O)(O)',
    'G': 'NC(C(=O)O)',
    'H': 'N[C@@H](C(=O)O)(CC1=CN(C=N1))',
    'I': 'N[C@@H](C(=O)O)C(C)CC',
    'L': 'N[C@@H](C(=O)O)CC(C)(C)',
    'K': 'N[C@@H](C(=O)O)CCCCN',
    'M': 'N[C@@H](C(=O)O)CCSC',
    'F': 'N[C@@H](C(=O)O)C(C1=cc=cc=C1)',
    'P': 'N1ccc[C@@H]1(C(=O)O)',
    'S': 'N[C@@H](C(=O)O)CO',
    'T': 'N[C@@H](C(=O)O)C(O)(C)',
    'W': 'N[C@@H](C(=O)O)(CC(C1=C2C=CC=C1)=CN2)',
    'Y': 'N[C@@H](C(=O)O)C(C1=cc=c(O)c=C1)',
    'V': 'N[C@@H](C(=O)O)C(C)(C)'
}

aa_vocab = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}


class PeptidePolicy(nn.Module):
    def __init__(self, aa_vocab_size=20, emb_dim=256, hidden_dim=512, nnAA_token_id=None):
        super().__init__()
        self.emb = nn.Embedding(aa_vocab_size + 1, emb_dim)
        self.nnAA_token_id = nnAA_token_id if nnAA_token_id is not None else aa_vocab_size
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.nat_head = nn.Linear(hidden_dim, aa_vocab_size)

        nn.init.constant_(self.gate_head.bias, -1.0)

    def forward(self, x_tokens, h=None):
        emb = self.emb(x_tokens)
        out, new_h = self.rnn(emb, h)
        gate_logits = self.gate_head(out)
        nat_logits = self.nat_head(out)
        return out, gate_logits, nat_logits, new_h


class Critic(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # 替换BatchNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h_final):
        return self.net(h_final)


class RLWrapper:
    def __init__(self, max_len):
        self.policy = PeptidePolicy().to(device)
        self.AA_generator = RNN(voc=tokenizer.vocab).to(device)
        self.AA_generator.load_state_dict(torch.load('Data\Model/Final.pth'))

        self.AA_generator_prior = copy.deepcopy(self.AA_generator).eval()
        for p in self.AA_generator_prior.parameters():
            p.requires_grad = False

        self.aa_vocab = aa_vocab
        self.max_len = max_len
        self.critic = Critic(hidden_dim=self.policy.rnn.hidden_size).to(device)

        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=5e-5)
        self.opt_sub = torch.optim.Adam(self.AA_generator.parameters(), lr=1e-4)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.entropy_coef_nat = 1e-3
        self.entropy_coef_gate = 5e-3
        self.entropy_coef_sub = 1e-4
        self.gate_usage_coef = -1e-3

        self.temperature = 1.0
        self.nnAA_id = self.policy.nnAA_token_id

    def sample_batch(self, batch_size, temperature, top_k, max_len, pattern="N[C@@H](C(=O)O)C*"):
        B = batch_size
        x = torch.zeros(B, 1, dtype=torch.long, device=device)
        h = None

        peptides = [[] for _ in range(B)]
        logprob_nat = torch.zeros(B, device=device)
        logprob_gate = torch.zeros(B, device=device)
        logprob_sub = torch.zeros(B, device=device)
        H_nat = torch.zeros(B, device=device)
        H_gate = torch.zeros(B, device=device)
        H_sub = torch.zeros(B, device=device)
        gate_usage = torch.zeros(B, device=device)
        used_sub_mask = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(self.max_len):
            max_retry = 20
            hiddens, gate_logits, nat_logits, h = self.policy(x, h)
            h_t = hiddens[:, -1, :]
            gate_logit_t = gate_logits[:, -1, 0]
            nat_logit_t = nat_logits[:, -1, :]

            gate_prob_t = torch.sigmoid(gate_logit_t)
            gate_dist = Bernoulli(probs=gate_prob_t.clamp(1e-6, 1 - 1e-6))
            g_t = gate_dist.sample()
            logprob_gate = logprob_gate + gate_dist.log_prob(g_t)
            H_gate = H_gate - (gate_prob_t * torch.log(gate_prob_t + 1e-12) +
                               (1 - gate_prob_t) * torch.log(1 - gate_prob_t + 1e-12))

            nat_logits_t = nat_logit_t / self.temperature
            nat_dist = Categorical(logits=nat_logits_t)
            aa_t = nat_dist.sample()
            logprob_nat = logprob_nat + nat_dist.log_prob(aa_t)
            H_nat = H_nat + nat_dist.entropy()

            next_x = torch.empty(B, dtype=torch.long, device=device)
            need_sub = (g_t > 0.5)

            # ======================== NNAA 子结构采样 ========================
            if need_sub.any():
                idxs = need_sub.nonzero(as_tuple=False).view(-1)
                sub_batch = idxs.size(0)

                valid_smiles_list = []
                valid_logp_tensors = []
                valid_H_tensors = []
                valid_prior_logp_tensors = []

                for i in range(sub_batch):
                    success = False
                    retry_count = 0

                    cur_smiles = ""
                    cur_logp = torch.zeros((), device=device)
                    cur_H = torch.zeros((), device=device)
                    cur_prior = torch.zeros((), device=device)

                    while not success and retry_count < max_retry:
                        seqs, sub_logp, sub_H = self.AA_generator.sample_pattern(
                            pattern=pattern, batch_size=1, max_len=max_len,
                            temperature=temperature, top_k=top_k
                        )
                        sub_logp0 = sub_logp.view(-1)[0]
                        sub_H0 = sub_H.view(-1)[0]

                        target = torch.stack(seqs, dim=0).to(device)
                        sub_logp_agent = self.AA_generator.likelihood(target)
                        sub_logp_agent0 = sub_logp_agent.view(-1)[0]
                        with torch.no_grad():
                            sub_logp_prior = self.AA_generator_prior.likelihood(target)
                        sub_logp_prior0 = sub_logp_prior.view(-1)[0]

                        cur_logp = sub_logp_agent0
                        cur_prior = sub_logp_prior0
                        seq = seqs[0]
                        pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g
                                in seq]
                        letters_string = ''.join(pred).replace('<s>', '').replace('</s>', '')
                        mol = Chem.MolFromSmiles(letters_string)

                        if mol is not None:
                            cur_smiles = letters_string
                            cur_logp = sub_logp_agent0
                            cur_H = sub_H0
                            cur_prior = sub_logp_prior0
                            success = True
                        else:
                            retry_count += 1

                    valid_smiles_list.append(cur_smiles)
                    valid_logp_tensors.append(cur_logp)
                    valid_H_tensors.append(cur_H)
                    valid_prior_logp_tensors.append(cur_prior)

                valid_logp_tensor = torch.stack(valid_logp_tensors)
                valid_H_tensor = torch.stack(valid_H_tensors)
                valid_prior_tensor = torch.stack(valid_prior_logp_tensors)

                used_sub_mask[idxs] = True

                tmp = torch.zeros_like(logprob_sub)
                tmp = tmp.index_add(0, idxs, valid_logp_tensor)
                logprob_sub = logprob_sub + tmp

                tmp_H = torch.zeros_like(H_sub)
                tmp_H = tmp_H.index_add(0, idxs, valid_H_tensor)
                H_sub = H_sub + tmp_H

                tmp_gate = torch.zeros_like(gate_usage)
                tmp_gate = tmp_gate.index_add(0, idxs, torch.ones_like(valid_H_tensor))
                gate_usage = gate_usage + tmp_gate

                if not hasattr(self, "logp_sub_prior_buffer"):
                    self.logp_sub_prior_buffer = []
                self.logp_sub_prior_buffer.append((idxs, valid_prior_tensor.detach()))

                for k, ii in enumerate(idxs.tolist()):
                    peptides[ii].append(("NNAA", valid_smiles_list[k]))
                    next_x[ii] = self.nnAA_id

            for i in (~need_sub).nonzero(as_tuple=False).view(-1).tolist():
                peptides[i].append(("AA", int(aa_t[i].item())))
                next_x[i] = aa_t[i]

            x = next_x.view(B, 1)

        h_final = h[-1]

        smiles_out = []
        for pep in peptides:
            smiles_list = []
            for kind, val in pep:
                if kind == "AA":
                    aa_letter = list(self.aa_vocab.keys())[list(self.aa_vocab.values()).index(val)]
                    smiles_list.append(Natural_AA[aa_letter])
                else:
                    smiles_list.append(val)
            smiles_out.append(process_smiles_chain(smiles_list) if smiles_list else "")

        if not hasattr(self, "logp_sub_prior_buffer"):
            self.logp_sub_prior_buffer = []

        logp_sub_prior_full = torch.zeros(B, device=device)
        for idxs, prior_vals in self.logp_sub_prior_buffer:
            logp_sub_prior_full = logp_sub_prior_full.index_add(0, idxs, prior_vals)
        self.logp_sub_prior_buffer = []

        return (
            smiles_out,
            peptides,
            logprob_nat,
            logprob_gate,
            logprob_sub,
            logp_sub_prior_full,
            H_nat,
            H_gate,
            H_sub,
            gate_usage,
            h_final,
            used_sub_mask
        )
