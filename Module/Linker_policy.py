import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_TYPES = ['disulfide', 'amide', 'ester', 'thioester', 'none', 'linker']


class CyclicPolicy(nn.Module):
    def __init__(self, fp_dim=2048, site_feat_dim=6, hidden=256, max_sites=30):
        super().__init__()
        self.max_sites = max_sites
        self.fp_proj = nn.Linear(fp_dim, hidden)
        self.site_enc = nn.Linear(site_feat_dim, hidden)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden*2 + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(ACTION_TYPES))
        )

    def forward(self, fp, site_feats):
        b = fp.shape[0]
        fp_h = F.relu(self.fp_proj(fp))
        n_sites = site_feats.shape[1]
        site_h = F.relu(self.site_enc(site_feats.view(-1, site_feats.shape[-1])))
        site_h = site_h.view(b, n_sites, -1)

        hi = site_h.unsqueeze(2).expand(-1, -1, n_sites, -1)
        hj = site_h.unsqueeze(1).expand(-1, n_sites, -1, -1)
        fp_ctx = fp_h.view(b, 1, 1, -1).expand(-1, n_sites, n_sites, -1)
        pair_input = torch.cat([hi, hj, fp_ctx], dim=-1)

        pair_logits = self.pair_mlp(pair_input)
        return pair_logits

    def sample_action(self, fp, site_feats, valid_pair_mask=None, temperature=1.0):
        logits = self.forward(fp, site_feats)
        b, n, _, n_types = logits.shape

        if valid_pair_mask is not None:
            logits = logits.masked_fill(~valid_pair_mask, -1e9)

        flat_logits = logits.view(b, -1) / max(temperature, 1e-6)
        probs = F.softmax(flat_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        chosen_flat = dist.sample()
        logp = dist.log_prob(chosen_flat)

        M = n * n * n_types
        t_idx = (chosen_flat % n_types).long()
        tmp = chosen_flat // n_types
        j_idx = (tmp % n).long()
        i_idx = (tmp // n).long()

        chosen_triplets = [(int(i_idx[k].item()), int(j_idx[k].item()), int(t_idx[k].item())) for k in range(b)]
        return chosen_flat, chosen_triplets, logp, logits
