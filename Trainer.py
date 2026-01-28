import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger
import openbabel as ob
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter

from Dataset_SMILES import DatasetSMILES, tokenizer
from Module.Module_Seq import RLWrapper, aa_vocab
from Module.Scoring_Module import calculate_model_score, ScoringFunction
from Module.FluoroScoring import ScoringFunction as FluoroScoring
from Module.FluoroScoring import calculate_fluro
from Module.Linker_policy import CyclicPolicy
from Module.Linker_cyclic_utils import (
    find_sites, mol_to_fp_tensor, build_site_feature_matrix,
    add_explicit_hydrogens, is_action_applicable
)
from Module.No_linker_cyclization import try_thioester, try_ester, try_amide, try_disulfide
from Module.Linker_cyclization import connect_linker_pep
from Module.Module_RNN import RNN

# -------------------------
# Global settings
# -------------------------
RDLogger.DisableLog('rdApp.*')
handler = ob.OBMessageHandler()
pybel.ob.obErrorLog.StopLogging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_TYPES = ['disulfide', 'amide', 'ester', 'thioester', 'none', 'linker']


def canonicalize_smiles(s: str):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def murcko_scaffold_smiles(s: str):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf is None:
            return None
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return None


def add_terminal_bromines_to_linker(smiles: str, require_two_sites: bool = True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)

    def carbon_can_add_single_bond(atom: Chem.Atom) -> bool:
        if atom.GetSymbol() != "C":
            return False
        if atom.GetFormalCharge() != 0:
            return False
        if atom.GetNumImplicitHs() > 0:
            return True
        try:
            ev = atom.GetExplicitValence()
            dv = atom.GetDefaultValence()
            if dv is None:
                return False
            return ev < dv
        except Exception:
            return False

    candidates = [a.GetIdx() for a in rw.GetAtoms() if carbon_can_add_single_bond(a)]
    if len(candidates) < 2:
        if not require_two_sites and len(candidates) == 1:
            c_idx = candidates[0]
            br_idx = rw.AddAtom(Chem.Atom("Br"))
            rw.AddBond(c_idx, br_idx, Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw)
                return rw.GetMol()
            except Exception:
                return None
        return None

    base_mol = rw.GetMol()

    best_pair = None
    best_dist = -1
    for i in range(len(candidates)):
        a = candidates[i]
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            path = rdmolops.GetShortestPath(base_mol, a, b)
            if not path:
                continue
            dist = len(path) - 1
            if dist > best_dist:
                best_dist = dist
                best_pair = (a, b)

    if best_pair is None:
        return None
    for c_idx in best_pair:
        br_idx = rw.AddAtom(Chem.Atom("Br"))
        rw.AddBond(c_idx, br_idx, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return rw.GetMol()


def add_terminal_CCBr_to_linker(smiles: str, require_two_sites: bool = True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)

    def carbon_can_add_single_bond(atom: Chem.Atom) -> bool:
        if atom.GetSymbol() != "C":
            return False
        if atom.GetFormalCharge() != 0:
            return False
        if atom.GetNumImplicitHs() > 0:
            return True
        try:
            ev = atom.GetExplicitValence()
            dv = atom.GetDefaultValence()
            if dv is None:
                return False
            return ev < dv
        except Exception:
            return False

    def add_CCBr_substituent(rw_mol: Chem.RWMol, attach_idx: int) -> None:
        c1 = Chem.Atom("C")
        c2 = Chem.Atom("C")
        br = Chem.Atom("Br")

        c1_idx = rw_mol.AddAtom(c1)
        c2_idx = rw_mol.AddAtom(c2)
        br_idx = rw_mol.AddAtom(br)

        rw_mol.AddBond(attach_idx, c1_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(c1_idx, c2_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(c2_idx, br_idx, Chem.BondType.SINGLE)

    candidates = [a.GetIdx() for a in rw.GetAtoms() if carbon_can_add_single_bond(a)]

    # 候选位点不足时：按 require_two_sites 决定加一个还是失败
    if len(candidates) < 2:
        if (not require_two_sites) and len(candidates) == 1:
            try:
                add_CCBr_substituent(rw, candidates[0])
                Chem.SanitizeMol(rw)
                return rw.GetMol()
            except Exception:
                return None
        return None

    base_mol = rw.GetMol()

    best_pair = None
    best_dist = -1
    for i in range(len(candidates)):
        a = candidates[i]
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            path = rdmolops.GetShortestPath(base_mol, a, b)
            if not path:
                continue
            dist = len(path) - 1
            if dist > best_dist:
                best_dist = dist
                best_pair = (a, b)

    if best_pair is None:
        return None
    try:
        for c_idx in best_pair:
            add_CCBr_substituent(rw, c_idx)
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


class TrainerRL:
    def __init__(self, max_len, save_dir=r"D:\PepBrick\Data\Result\generation\new"):
        self.model = RLWrapper(max_len=max_len)

        self.cyc_policy = CyclicPolicy(fp_dim=2048, site_feat_dim=6, hidden=256).to(device)
        self.opt_cyc_policy = torch.optim.Adam(self.cyc_policy.parameters(), lr=1e-4)

        self.voc = tokenizer.vocab
        self.linker_generator_prior = RNN(self.voc).to(device)
        self.linker_generator_agent = RNN(self.voc).to(device)

        self.linker_generator_prior.load_state_dict(torch.load('Data\Model/Final.pth'))
        self.linker_generator_prior.eval()
        self.linker_generator_agent.load_state_dict(torch.load('Data\Model/Final.pth'))
        self.opt_lg = torch.optim.Adam(self.linker_generator_agent.parameters(), lr=3e-4)

        self.scoring_model = ScoringFunction(name='Cyclic_reg', func='regression')
        self.scoring_model_affi = ScoringFunction(name='Opioid', func='regression')
        self.scoring_model_neuro = ScoringFunction(name='Neuro', func='classification')

        self.contribution_model = AASmilesGRUModel(rl_tokenizer).to(device)
        self.contribution_model.load_state_dict(
            torch.load('D:/PepBrick/Monomer/Data/Model/AA_contribution_pretrain_2.pth', map_location=device)
        )
        self.contribution_model.eval()
        self.contribution_tokenizer = AutoTokenizer.from_pretrained(
            "D:/PepBrick/Data/Dataset/SMILES", ignore_mismatched_sizes=False
        )
        self.idx_to_aa = {v: k for k, v in aa_vocab.items()}
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.model.entropy_coef_nat = 1e-3
        self.model.entropy_coef_gate = 1e-2
        self.model.entropy_coef_sub = 1e-3
        self.best_avg_reward = -999
        self.sigma_sub = 15
        self.sigma_linker = 60
        self.grad_clip = 0.5
        self.cyc_entropy_coef = 0.1
        self.sequence_temperature = 1.5
        self.sequence_top_k = 20
        self.sequence_max_len = 100
        self.sequence_sample_pattern = "N[C@@H](C(=O)O)*"
        self.linker_entropy_coef = 0
        self.linker_sample_pattern = "C*"
        self.linker_max_attempts = 10
        self.linker_temperature = 1.5
        self.linker_top_k = 20
        self.linker_max_len = 100
        self.enforce_unique_in_step = True
        self.unique_key = "canonical"
        self.resample_chunk = 32
        self.max_resample_rounds = 30
        self.duplicate_reward_penalty = 0.0
        self.idx_to_aa = {v: k for k, v in aa_vocab.items()}
        self.none_action_idx = ACTION_TYPES.index('none')
        self.tN = len(ACTION_TYPES)

    def sample_unique_batch(self, target_batch_size: int):
        assert target_batch_size > 0

        def get_key(sm: str):
            if self.unique_key == "scaffold":
                return murcko_scaffold_smiles(sm) or canonicalize_smiles(sm) or sm
            return canonicalize_smiles(sm) or sm

        collected_smiles = []
        collected_peptides = []
        collected_logp_nat = []
        collected_logp_gate = []
        collected_logp_sub_agent = []
        collected_logp_sub_prior = []
        collected_H_nat = []
        collected_H_gate = []
        collected_H_sub = []
        collected_gate_usage = []
        collected_h_final = []
        collected_used_sub_mask = []

        seen = set()
        rounds = 0

        while len(collected_smiles) < target_batch_size and rounds < self.max_resample_rounds:
            rounds += 1
            need = target_batch_size - len(collected_smiles)
            sample_n = max(need, self.resample_chunk)

            (smiles_out, peptides,
             logp_nat, logp_gate, logp_sub_agent,
             logp_sub_prior,
             H_nat, H_gate, H_sub, gate_usage, h_final, used_sub_mask) = self.model.sample_batch(
                batch_size=sample_n,
                temperature=self.sequence_temperature,
                top_k=self.sequence_top_k,
                pattern=self.sequence_sample_pattern,
                max_len=self.sequence_max_len
            )

            if not isinstance(used_sub_mask, torch.Tensor):
                used_sub_mask = torch.tensor(used_sub_mask, device=h_final.device)

            for i, sm in enumerate(smiles_out):
                key = get_key(sm)
                if key in seen:
                    continue
                seen.add(key)

                collected_smiles.append(sm)
                collected_peptides.append(peptides[i])

                collected_logp_nat.append(logp_nat[i])
                collected_logp_gate.append(logp_gate[i])
                collected_logp_sub_agent.append(logp_sub_agent[i])
                collected_logp_sub_prior.append(logp_sub_prior[i])

                collected_H_nat.append(H_nat[i])
                collected_H_gate.append(H_gate[i])
                collected_H_sub.append(H_sub[i])

                collected_gate_usage.append(gate_usage[i])
                collected_h_final.append(h_final[i])
                collected_used_sub_mask.append(used_sub_mask[i])

                if len(collected_smiles) >= target_batch_size:
                    break
        if len(collected_smiles) < target_batch_size:
            print(f"[WARN] unique sampling got {len(collected_smiles)}/{target_batch_size} "
                  f"after {rounds} rounds. Consider increasing temperature/top_k or use canonical uniqueness.")
        if len(collected_smiles) == 0:
            raise RuntimeError("Unique sampling returned empty batch.")
        def stack_list(lst):
            return torch.stack([x if torch.is_tensor(x) else torch.tensor(x, device=device) for x in lst], dim=0)
        smiles_out = collected_smiles
        peptides = collected_peptides
        logp_nat = stack_list(collected_logp_nat)
        logp_gate = stack_list(collected_logp_gate)
        logp_sub_agent = stack_list(collected_logp_sub_agent)
        logp_sub_prior = stack_list(collected_logp_sub_prior)
        H_nat = stack_list(collected_H_nat)
        H_gate = stack_list(collected_H_gate)
        H_sub = stack_list(collected_H_sub)
        gate_usage = stack_list(collected_gate_usage)
        h_final = torch.stack(collected_h_final, dim=0)
        used_sub_mask = torch.stack(collected_used_sub_mask, dim=0)

        return (smiles_out, peptides,
                logp_nat, logp_gate, logp_sub_agent,
                logp_sub_prior,
                H_nat, H_gate, H_sub, gate_usage, h_final, used_sub_mask)

    def process_with_bond_policy(self, smiles_list):
        if not isinstance(smiles_list, list):
            smiles_list = list(smiles_list)

        batch_size = len(smiles_list)

        batch_actions = [None] * batch_size
        batch_log_probs = [None] * batch_size
        batch_logits = [None] * batch_size
        batch_new_smiles = [None] * batch_size
        failed_indices = []
        cyc_type_list = []
        linker_smile_list = [None] * batch_size

        linker_indices = []
        linker_logps_agent = []
        linker_logps_prior = []
        linker_entropies = []
        linker_attempts_info = []

        fps = []
        site_feats_list = []
        sites_list = []
        mols_list = []

        for idx in range(batch_size):
            smile = smiles_list[idx]
            try:
                smiles_h = add_explicit_hydrogens(smile)
                mol, sites = find_sites(smiles_h)
                if mol is None:
                    raise ValueError(f"Invalid molecule for SMILES: {smiles_h}")

                fp = mol_to_fp_tensor(mol)
                site_feats = build_site_feature_matrix(mol, sites, max_sites=self.cyc_policy.max_sites)
                fps.append(fp)
                site_feats_list.append(site_feats)
                sites_list.append(sites)
                mols_list.append((mol, smiles_h))
            except Exception:
                failed_indices.append(idx)
                fps.append(torch.zeros((1, 2048), dtype=torch.float32))
                site_feats_list.append(torch.zeros((1, self.cyc_policy.max_sites, 6), dtype=torch.float32))
                sites_list.append([])
                mols_list.append((None, smile))

        fps = torch.cat(fps, dim=0).to(device, non_blocking=True)
        site_feats = torch.cat(site_feats_list, dim=0).to(device, non_blocking=True)

        b = batch_size
        n = self.cyc_policy.max_sites
        valid_mask_cpu = torch.zeros((b, n, n, self.tN), dtype=torch.bool)

        for bi in range(b):
            sites = sites_list[bi]
            n_sites = len(sites)

            for i in range(n_sites):
                si = sites[i]
                for j in range(n_sites):
                    if i == j:
                        continue
                    sj = sites[j]
                    for t_idx, atype in enumerate(ACTION_TYPES):
                        if is_action_applicable(si, sj, atype):
                            valid_mask_cpu[bi, i, j, t_idx] = True

            if n_sites > 0:
                valid_mask_cpu[bi, :n_sites, :n_sites, self.none_action_idx] = True

        valid_mask = valid_mask_cpu.to(device, non_blocking=True)

        chosen_flat, chosen_triplets, logp, logit = self.cyc_policy.sample_action(
            fps, site_feats, valid_pair_mask=valid_mask
        )

        for idx in range(batch_size):
            try:
                i_idx, j_idx, t_idx = chosen_triplets[idx]
                cyc_type_list.append(ACTION_TYPES[t_idx])

                mol, smiles_h = mols_list[idx]
                if mol is None:
                    batch_new_smiles[idx] = smiles_list[idx]
                    batch_log_probs[idx] = torch.tensor(0.0, device=device)
                    continue

                try:
                    batch_logits[idx] = logit[idx]
                except Exception:
                    batch_logits[idx] = None

                si = sites_list[idx][i_idx]
                sj = sites_list[idx][j_idx]
                new_smile = smiles_list[idx]

                act = ACTION_TYPES[t_idx]
                if act == 'disulfide':
                    new_smile = try_disulfide(smiles_h, si['atom_idx'], sj['atom_idx'])

                elif act == 'amide':
                    c_idx = si['atom_idx'] if si['type'] == 'carboxyl' else sj['atom_idx']
                    n_idx = sj['atom_idx'] if si['type'] == 'carboxyl' else si['atom_idx']
                    new_smile = try_amide(smiles_h, c_idx, n_idx)

                elif act == 'ester':
                    c_idx = si['atom_idx'] if si['type'] == 'carboxyl' else sj['atom_idx']
                    o_idx = sj['atom_idx'] if si['type'] == 'carboxyl' else si['atom_idx']
                    new_smile = try_ester(smiles_h, c_idx, o_idx)

                elif act == 'thioester':
                    s_idx = si['atom_idx'] if si['type'] == 'thiol' else sj['atom_idx']
                    c_idx = sj['atom_idx'] if si['type'] == 'thiol' else si['atom_idx']
                    new_smile = try_thioester(smiles_h, c_idx, s_idx)

                elif act == 'linker':
                    last_raw_linker = None
                    for attempt in range(self.linker_max_attempts):
                        seqs, _, ent = self.linker_generator_agent.sample_pattern(
                            batch_size=1,
                            pattern=self.linker_sample_pattern,
                            temperature=self.linker_temperature,
                            top_k=self.linker_top_k
                        )

                        seq = seqs[0]
                        raw_linker_smiles = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                        raw_linker_smiles = raw_linker_smiles.replace("<s>", "").replace("</s>", "").replace(" ", "")
                        last_raw_linker = raw_linker_smiles

                        linker_mol_with_br = add_terminal_CCBr_to_linker(raw_linker_smiles)
                        if linker_mol_with_br is None:
                            continue
                        linker_smiles_with_br = Chem.MolToSmiles(linker_mol_with_br)

                        try:
                            connected = connect_linker_pep(
                                smiles_h,
                                linker_smiles_with_br,
                                si['atom_idx'],
                                sj['atom_idx']
                            )
                            if connected is not None:
                                new_smile = connected
                                with torch.no_grad():
                                    target = seq
                                    logps_prior = self.linker_generator_prior.likelihood(target.unsqueeze(0))
                                    logps_agent = self.linker_generator_agent.likelihood(target.unsqueeze(0))

                                linker_indices.append(idx)
                                linker_logps_agent.append(logps_agent.view(-1)[0].to(device))
                                linker_logps_prior.append(logps_prior.view(-1)[0].to(device))
                                linker_entropies.append(ent.view(-1)[0].to(device))
                                linker_attempts_info.append(attempt + 1)
                                break
                        except Exception:
                            continue

                    linker_smile_list[idx] = last_raw_linker

                batch_new_smiles[idx] = new_smile
                batch_log_probs[idx] = logp[idx]

            except Exception:
                failed_indices.append(idx)
                batch_new_smiles[idx] = smiles_list[idx]
                batch_log_probs[idx] = torch.tensor(0.0, device=device)
                linker_smile_list[idx] = None

        linker_logps_agent_tensor = torch.stack(linker_logps_agent).to(device) if linker_logps_agent else None
        linker_logps_prior_tensor = torch.stack(linker_logps_prior).to(device) if linker_logps_prior else None
        linker_entropies_tensor = torch.stack(linker_entropies).to(device) if linker_entropies else None

        return (batch_actions, batch_log_probs, batch_logits, batch_new_smiles,
                failed_indices, cyc_type_list,
                linker_indices, linker_logps_agent_tensor, linker_logps_prior_tensor, linker_entropies_tensor,
                linker_attempts_info, linker_smile_list)

    def extract_aa_nnaa_smiles(self, peptides):
        aa_nnaa_smiles_list = []
        nnaa_indices_list = []
        for pep_seq in peptides:
            pos_smiles = []
            nnaa_indices = []
            for j, (tag, value) in enumerate(pep_seq):
                if tag == "AA":
                    aa_letter = self.idx_to_aa.get(value, "UNK")
                    pos_smiles.append(Natural_AA.get(aa_letter, ""))
                elif tag == "NNAA":
                    pos_smiles.append(str(value).strip())
                    nnaa_indices.append(j)
                else:
                    pos_smiles.append("")
            aa_nnaa_smiles_list.append(pos_smiles)
            nnaa_indices_list.append(nnaa_indices)
        return aa_nnaa_smiles_list, nnaa_indices_list

    def prepare_contribution_input(self, aa_nnaa_smiles_list):
        if not aa_nnaa_smiles_list:
            return None, None, 0

        max_aa_length = max(len(smiles_list) for smiles_list in aa_nnaa_smiles_list)
        max_smiles_length = 100
        pad_token_id = self.contribution_tokenizer.vocab.get(self.contribution_tokenizer.pad_token, 0)

        input_ids_list = []
        attention_mask_list = []

        for smiles_list in aa_nnaa_smiles_list:
            pos_input_ids = []
            pos_attention_mask = []
            for smiles in smiles_list:
                tokens = self.contribution_tokenizer.tokenize(smiles) if smiles else []
                if len(tokens) < max_smiles_length:
                    tokens += [self.contribution_tokenizer.pad_token] * (max_smiles_length - len(tokens))
                else:
                    tokens = tokens[:max_smiles_length]

                ids = self.contribution_tokenizer.convert_tokens_to_ids(tokens)
                if len(ids) < max_smiles_length:
                    ids += [pad_token_id] * (max_smiles_length - len(ids))

                valid_len = len([t for t in tokens if t != self.contribution_tokenizer.pad_token])
                mask = [1] * valid_len + [0] * (max_smiles_length - valid_len)

                pos_input_ids.append(ids)
                pos_attention_mask.append(mask)

            num_pos = len(pos_input_ids)
            if num_pos < max_aa_length:
                pad_ids = [pad_token_id] * max_smiles_length
                pad_mask = [0] * max_smiles_length
                for _ in range(max_aa_length - num_pos):
                    pos_input_ids.append(pad_ids)
                    pos_attention_mask.append(pad_mask)

            input_ids_list.append(pos_input_ids)
            attention_mask_list.append(pos_attention_mask)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=device)
        return input_ids, attention_mask, max_aa_length

    def compute_contribution_rewards(self, aa_nnaa_smiles_list):
        batch_size = len(aa_nnaa_smiles_list)
        if batch_size == 0:
            return torch.zeros((0, 0), device=device), []

        input_ids, attention_mask, max_aa_length = self.prepare_contribution_input(aa_nnaa_smiles_list)
        if input_ids is None:
            return torch.zeros((batch_size, max_aa_length), device=device), [[] for _ in range(batch_size)]

        with torch.no_grad():
            _, contributions, _ = self.contribution_model(input_ids, attention_mask)
            contributions = 2 * contributions - 1  # -> [-1, 1]
        contributions = contributions.to(device)
        valid_pos_mask = (attention_mask.sum(dim=2) > 0).float()  # (B, L)
        contribution_rewards = contributions * valid_pos_mask

        all_contributions = contribution_rewards.detach().cpu().tolist()
        return contribution_rewards, all_contributions

    def train(self, iteration=100, batch_size=32, save_every=1, debug_anomaly=False):
        torch.autograd.set_detect_anomaly(bool(debug_anomaly))

        reward_history = []

        for step in tqdm(range(1, iteration + 1)):
            self.model.policy.train()
            self.model.AA_generator.train()
            self.model.critic.train()
            self.cyc_policy.train()
            self.linker_generator_agent.train()

            if self.enforce_unique_in_step:
                (smiles_out, peptides,
                 logp_nat, logp_gate, logp_sub_agent,
                 logp_sub_prior,
                 H_nat, H_gate, H_sub, gate_usage, h_final, used_sub_mask) = self.sample_unique_batch(batch_size)
            else:
                (smiles_out, peptides,
                 logp_nat, logp_gate, logp_sub_agent,
                 logp_sub_prior,
                 H_nat, H_gate, H_sub, gate_usage, h_final, used_sub_mask) = self.model.sample_batch(
                    batch_size=batch_size,
                    temperature=self.sequence_temperature,
                    top_k=self.sequence_top_k,
                    pattern=self.sequence_sample_pattern,
                    max_len=self.sequence_max_len
                )

            B = len(smiles_out)

            used_sub_mask = used_sub_mask.to(device) if isinstance(used_sub_mask, torch.Tensor) else torch.tensor(
                used_sub_mask, device=device)

            (batch_actions, batch_log_probs, batch_logits, batch_new_smiles,
             failed_indices, cyc_type_list,
             linker_indices, linker_logps_agent_tensor, linker_logps_prior_tensor, linker_entropies_tensor,
             linker_attempts_info, linker_smile_list) = self.process_with_bond_policy(smiles_out)

            with torch.inference_mode():
                all_smiles = list(smiles_out) + list(batch_new_smiles)
                logp_all, true_all, valid_all = calculate_model_score(
                    all_smiles, scoring_model=self.scoring_model
                )
                affi_logp_all, affi_true_all, _ = calculate_model_score(
                    all_smiles, scoring_model=self.scoring_model_affi
                )
                neur_logp_all, neur_true_all, _ = calculate_model_score(
                    all_smiles, scoring_model=self.scoring_model_neuro
                )

                logp_before_cycle = logp_all[:B]
                logp_after_cycle = logp_all[B:]
                true_logP_before_cycle = true_all[:B]
                true_logP_after_cycle = true_all[B:]
                affi_logp_before_cycle = affi_logp_all[:B]
                affi_logp_after_cycle = affi_logp_all[B:]
                affi_true_logP_before_cycle = affi_true_all[:B]
                affi_true_logP_after_cycle = affi_true_all[B:]
                neur_logp_before_cycle = neur_logp_all[:B]
                neur_logp_after_cycle = neur_logp_all[B:]
                neur_true_logP_before_cycle = neur_true_all[:B]
                neur_true_logP_after_cycle = neur_true_all[B:]
                valid_mask_before = valid_all[:B]

            aa_nnaa_smiles_list, nnaa_indices_list = self.extract_aa_nnaa_smiles(peptides)
            contribution_rewards, all_contributions = self.compute_contribution_rewards(aa_nnaa_smiles_list)
            has_nnaa = used_sub_mask.clone()
            valid_nnaa = (has_nnaa & valid_mask_before).clone()
            nnaa_bonus = torch.where(
                valid_nnaa,
                torch.tensor(0.1, device=device),
                torch.tensor(0.0, device=device)
            )

            reward_before = logp_before_cycle*0.7 + affi_logp_before_cycle*0.3 + neur_logp_before_cycle*0.1 + nnaa_bonus
            reward_after = logp_after_cycle*0.7 + affi_logp_after_cycle*0.3 + neur_logp_after_cycle*0.1 + nnaa_bonus
            delta_reward_cyc = reward_after - reward_before

            if self.duplicate_reward_penalty > 0:
                keys = [canonicalize_smiles(s) or s for s in smiles_out]
                c = Counter(keys)
                dup_mask = torch.tensor([1.0 if c[k] > 1 else 0.0 for k in keys], device=device)
                reward_before = reward_before - self.duplicate_reward_penalty * dup_mask
                reward_after = reward_after - self.duplicate_reward_penalty * dup_mask

            valid_mask = torch.ones(B, dtype=torch.bool, device=device)
            for idx in failed_indices:
                if idx < B:
                    valid_mask[idx] = False
            valid_mask = valid_mask & (logp_before_cycle != -0.2) & (logp_after_cycle != -0.2)

            reward_history.append(reward_before.mean().item())
            if len(reward_history) > 100:
                reward_history = reward_history[1:]

            V = self.model.critic(h_final.detach()).squeeze(1)  # (B,)
            advantage = (reward_before - V.detach()).clone()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            avg_gate_usage = (gate_usage / float(self.model.max_len)).mean()
            gate_explore_reward = 0.1 * avg_gate_usage

            policy_obj = (logp_gate + logp_nat + logp_sub_agent.detach())
            policy_loss = -(advantage * policy_obj).mean()

            ent_loss = -(self.model.entropy_coef_gate * H_gate.mean()
                         + self.model.entropy_coef_nat * H_nat.mean()
                         + self.model.entropy_coef_sub * H_sub.mean())

            total_policy_loss = policy_loss + ent_loss - gate_explore_reward

            self.model.opt_policy.zero_grad(set_to_none=True)
            total_policy_loss.backward()  # FIX: 不要 retain_graph=True
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.grad_clip)
            self.model.opt_policy.step()

            self.model.opt_sub.zero_grad(set_to_none=True)
            sub_loss = torch.tensor(0.0, device=device)

            if has_nnaa.any():
                contrib_signal = (nnaa_contrib_per_sample - nnaa_contrib_per_sample.mean()).detach()
                r_sub = (reward_before - reward_before.mean()).detach()
                r_sub = r_sub*0.5 + contrib_signal*0.5
                r_sub = r_sub.clamp(-1.0, 1.0)

                aug_logp = logp_sub_prior + self.sigma_sub * r_sub
                target = aug_logp[has_nnaa].detach()
                pred = logp_sub_agent[has_nnaa]
                if target.numel() > 0:
                    sub_loss = F.smooth_l1_loss(pred, target, reduction='mean')

            if float(sub_loss.item()) > 0.0:
                sub_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.AA_generator.parameters(), self.grad_clip)
                self.model.opt_sub.step()

            V_new = self.model.critic(h_final.detach()).squeeze(1)
            self.model.opt_critic.zero_grad(set_to_none=True)
            critic_loss = F.mse_loss(V_new, reward_before.detach())
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.grad_clip)
            self.model.opt_critic.step()

            bp = batch_log_probs
            has_logp_mask = torch.tensor([x is not None for x in bp], device=device, dtype=torch.bool)
            cyc_valid_mask = valid_mask & has_logp_mask

            if cyc_valid_mask.any():
                idxs = cyc_valid_mask.nonzero(as_tuple=False).view(-1).tolist()
                cyc_logps_tensor = torch.stack([bp[i] for i in idxs])

                valid_delta_reward = delta_reward_cyc[cyc_valid_mask]
                valid_delta_reward = (valid_delta_reward - valid_delta_reward.mean()) / (valid_delta_reward.std() + 1e-8)
                cyc_policy_loss = -(valid_delta_reward * cyc_logps_tensor).mean()

                valid_logits_list = []
                for i in idxs:
                    if batch_logits[i] is not None:
                        valid_logits_list.append(batch_logits[i].to(device))

                if valid_logits_list:
                    logits_cat = torch.stack(valid_logits_list).to(device)
                    flat = logits_cat.view(logits_cat.shape[0], -1)
                    probs = F.softmax(flat, dim=-1)
                    ent = - (probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    cyc_ent_loss = -self.cyc_entropy_coef * ent
                else:
                    cyc_ent_loss = torch.tensor(0.0, device=device)

                total_cyc_loss = cyc_policy_loss + cyc_ent_loss
                self.opt_cyc_policy.zero_grad(set_to_none=True)
                total_cyc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cyc_policy.parameters(), 1.0)
                self.opt_cyc_policy.step()

            self.opt_lg.zero_grad(set_to_none=True)
            linker_loss = torch.tensor(0.0, device=device)

            if (linker_logps_agent_tensor is not None) and (linker_logps_prior_tensor is not None) and (
                    len(linker_indices) > 0):
                idx_t = torch.tensor(linker_indices, device=device, dtype=torch.long)

                linker_valid_for_update = valid_mask[idx_t]

                if linker_valid_for_update.any():
                    keep = linker_valid_for_update.nonzero(as_tuple=False).view(-1)

                    agent_logp = linker_logps_agent_tensor[keep]
                    prior_logp = linker_logps_prior_tensor[keep]

                    reward = delta_reward_cyc[idx_t][keep]

                    reward = reward.clamp(-1.0, 1.0)

                    augmented_logp = prior_logp + self.sigma_linker * reward
                    base_loss = F.smooth_l1_loss(agent_logp, augmented_logp.detach(), reduction='mean')
                    linker_loss = base_loss

                    if linker_entropies_tensor is not None:
                        ent = linker_entropies_tensor[keep].mean()
                        linker_loss = linker_loss - self.linker_entropy_coef * ent

                    self.opt_lg.zero_grad(set_to_none=True)
                    linker_loss.backward()
                    self.opt_lg.step()

            if (step % max(int(save_every), 1)) == 0:
                max_len_pep = max(len(p) for p in peptides) if len(peptides) > 0 else 0
                peptide_dict = {}
                for i in range(max_len_pep):
                    peptide_dict[f"Position_{i + 1}"] = []
                    peptide_dict[f"NNAA_{i + 1}"] = []
                    peptide_dict[f"Contribution_{i + 1}"] = []

                for idx, pep_seq in enumerate(peptides):
                    sample_contrib = all_contributions[idx] if idx < len(all_contributions) else []
                    for i in range(max_len_pep):
                        if i < len(pep_seq):
                            tag, value = pep_seq[i]
                            if tag == "AA":
                                aa_letter = self.idx_to_aa.get(value, "UNK")
                                peptide_dict[f"Position_{i + 1}"].append(aa_letter)
                                peptide_dict[f"NNAA_{i + 1}"].append("")
                            elif tag == "NNAA":
                                peptide_dict[f"Position_{i + 1}"].append("NNAA")
                                peptide_dict[f"NNAA_{i + 1}"].append(value)
                            else:
                                peptide_dict[f"Position_{i + 1}"].append("UNK")
                                peptide_dict[f"NNAA_{i + 1}"].append("")

                            try:
                                peptide_dict[f"Contribution_{i + 1}"].append(
                                    round(sample_contrib[i], 6) if i < len(sample_contrib) else ""
                                )
                            except Exception:
                                peptide_dict[f"Contribution_{i + 1}"].append("")
                        else:
                            peptide_dict[f"Position_{i + 1}"].append("")
                            peptide_dict[f"NNAA_{i + 1}"].append("")
                            peptide_dict[f"Contribution_{i + 1}"].append("")

                has_nnaa_list = has_nnaa.detach().cpu().numpy().tolist()
                valid_list = valid_mask.detach().cpu().numpy().tolist()

                if len(cyc_type_list) < B:
                    cyc_type_list += ["none"] * (B - len(cyc_type_list))

                df_base = pd.DataFrame({
                    "SMILES": smiles_out,
                    "SMILES_cyc": batch_new_smiles,
                    "SMILES_linker": linker_smile_list,
                    "Has_NNAA": has_nnaa_list,
                    "Is_Valid": valid_list,
                    "cyc_type": cyc_type_list,

                    "True_logP_Before": true_logP_before_cycle.detach().cpu().numpy(),
                    "True_logP_After": true_logP_after_cycle.detach().cpu().numpy(),
                    "True_affi_Before": affi_true_logP_before_cycle.detach().cpu().numpy(),
                    "True_affi_After": affi_true_logP_after_cycle.detach().cpu().numpy(),
                    "True_neur_Before": neur_true_logP_before_cycle.detach().cpu().numpy(),
                    "True_neur_After": neur_true_logP_after_cycle.detach().cpu().numpy(),
                    "Total_Reward": reward_after.detach().cpu().numpy(),
                    "Total_Reward_delta": delta_reward_cyc.detach().cpu().numpy(),

                })

                df_peptides = pd.DataFrame(peptide_dict)
                df = pd.concat([df_base, df_peptides], axis=1)

                df.to_csv(os.path.join(self.save_dir, f"step_{step}.csv"), index=False)

            valid_rate = float(valid_mask.float().mean().item())
            gate_rate = float(has_nnaa.float().mean().item())

            mean_reward_before = float(reward_before.mean().item())
            mean_reward_after = float(reward_after.mean().item())

            print(f"\n[Step {step}] "
                  f"PolicyLoss={total_policy_loss.item():.4f} | "
                  f"SubLoss={sub_loss.item():.4f} | "
                  f"CriticLoss={critic_loss.item():.4f} | "
                  f"LinkerLoss={linker_loss.item():.4f} | "
                  f"MeanReward_before={mean_reward_before:.4f} | "
                  f"MeanReward_after={mean_reward_after:.4f} | "
                  f"ValidRate={valid_rate:.3f} | "
                  f"NNAA_Rate={gate_rate:.3f} | "
                  f"Temp={self.model.temperature:.3f}")


if __name__ == "__main__":
    trainer = TrainerRL(max_len=6, save_dir="Data/Result/generation/new")
    trainer.train(iteration=100, batch_size=32, save_every=1, debug_anomaly=False)


