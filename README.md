# 🧪 De Novo Design of Peptidomimetics via Hierarchical Coupling Reinforcement Learning

---

## 📄 Abstract
> **Abstract**
<p align="center">
  <img src="TOC.png" alt="Overview Figure" width="700"/>
</p>
Peptidomimetic design is fundamentally constrained by the strong coupling among noncanonical residues, sequence composition, and molecular topology within rugged, multi-objective fitness landscapes. To address this challenge, we developed PepBrick, a structured multi-level nested reinforcement learning framework that hierarchically represents peptidomimetics and decomposes the design process into controllable decision steps. By enabling coordinated optimization of chemical composition and molecular topology under multiple constraints, PepBrick provides a flexible strategy for navigating complex peptidomimetic design spaces. Systematic evaluation at both the component and integrated levels demonstrated its robust design capability. Moreover, the de novo design of blood-brain-barrier-penetrating neuropeptidomimeticss and tumor-targeting fluorescent probes, together with proof-of-concept molecular dynamics simulations, further demonstrated its effectiveness in navigating strongly coupled, multi-objective fitness landscapes. Collectively, these results establish PepBrick as a general computational framework for programmable peptidomimetic design and the development of next-generation peptide-based functional molecules.

---

## 🖼️ Overview
> **The PepBrick framework：** (i) Sequence Generation Module: Selects canonical amino acids from the candidate pool or activates the unnatural amino acid generation module when needed. (iii) Amino Acid Generation Module: Activated by the sequence generation module to produce diverse unnatural amino acid structures. (ii) Cyclization Policy Module: Determines the cyclization strategy and sites for linear peptidomimetics and decides whether to activate the linker generator. (iv) Linker Generation Module: Activated by the cyclization policy module to generate structurally feasible linker molecules. (v) Scoring Module: Incorporates a global-level graph neural network encoder, an amino-acid-level language model, and a Critic network to enhance training stability. This module outputs the reward signal for generated peptidomimetics and updates the parameters of the entire PepBrick framework.

<p align="center">
  <img src="Fig1.png" alt="Overview Figure" width="700"/>
</p>


---

## ⚙️ Environment Requirements

- **Python**: >= 3.8
- **CUDA** : >= 12.6
- **Key Dependencies**:
  ```text
  torch = 2.5.1
  numpy = 1.26.4
  pandas = 2.3.3
  scikit-learn = 1.6.1
  rdkit = 2025.03.5
  openbabel = 3.1.1
  torch_geometric = 2.6.1


## ⚙️ Data preparation, scoring, and reinforcement learning


### 1. Environment and example files

Create an environment and install the packages listed in the repository's environment requirements. For example, the following installs a CPU build for the scoring examples:

```bash
conda create -n pepbrick python=3.10
conda activate pepbrick
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 pandas==2.3.3 scikit-learn==1.6.1 rdkit==2025.3.5 torch-geometric==2.6.1 tqdm
```

The generation workflow additionally imports `transformers` and Open Babel:

```bash
python -m pip install transformers
conda install -c conda-forge openbabel=3.1.1
```

For GPU runs, use a CUDA-enabled PyTorch build compatible with the machine and record its exact version. Record the resolved environment for each experiment:

### 2. Required scoring data

Prepare the cleaned, final CSV partitions used in the study. Do not generate a new random split when attempting to reproduce a reported result.

| Target | Input columns |
|---|---|
| CPP identification | `SMILES`, `Label` |
| Neuropeptide identification | `SMILES`, `Label` |
| Membrane permeability classification or regression | `SMILES`, `Label` |
| MOR docking-score regression | `SMILES`, `Label` |
| GRPR docking-score regression | `SMILES`, `Label` |

 Docking-score labels are computational scores rather than experimentally measured affinities. 

Example CSV structure:

```csv
SMILES,Label
NCC(=O)O,1
N[C@@H](C)C(=O)O,0
```

These rows illustrate the schema only; they are not a training dataset. Sequence-based datasets must first be converted to their intended molecular structures. The graph converter expects the final molecular SMILES and does not infer modifications or cyclization from sequence strings.


### 3. Convert CSV data into molecular graphs

Molecular graphs are prepared using `Graph_dataset.py`. The input CSV path specifies the source records, while `root` specifies the directory in which the processed graph dataset is stored.

The current `read_data_labeled()` function reads molecular SMILES from `SMILES_cyc` and labels from `affinity`. If the CSV uses different column names, update these names in `read_data_labeled()` accordingly.

Remove the original module-level input-loading statements:

```python
path = r'Data\Dataset'
labeled_data = read_data_labeled(path)
```

Then append the following code after the `DatasetWithLabel` class definition in `Graph_dataset.py`:

```python
if __name__ == "__main__":
    from pathlib import Path

    csv_dir = Path("Data/Dataset/NeuroPep")
    graph_dir = Path("Data/Dataset/graph_data")

    for split in ["train", "valid", "test"]:
        path = str(csv_dir / f"{split}.csv")
        labeled_data = read_data_labeled(path)

        root = str(graph_dir / f"Neuro_{split}")
        dataset = DatasetWithLabel(root=root)

        print(f"{split}: {len(dataset)} graphs")
        print(f"Output: {dataset.processed_paths[0]}")
```

Run the conversion from the repository root:

```bash
python "GNN Scoring/Graph_dataset.py"
```

For each record, RDKit parses the SMILES into a molecule, and `mol_to_graph_data()` constructs its graph representation. The molecular label is stored as `data.label`. The resulting graphs are collated and saved as:

```text
Data/Dataset/graph_data/Neuro_train/processed/processed_data.dataset
Data/Dataset/graph_data/Neuro_valid/processed/processed_data.dataset
Data/Dataset/graph_data/Neuro_test/processed/processed_data.dataset
```

The scoring-model training code can subsequently load these prepared datasets by specifying their respective roots:

```python
train_dataset = DatasetWithLabel(
    root="Data/Dataset/graph_data/Neuro_train"
)
valid_dataset = DatasetWithLabel(
    root="Data/Dataset/graph_data/Neuro_valid"
)
test_dataset = DatasetWithLabel(
    root="Data/Dataset/graph_data/Neuro_test"
)
```

Use a separate root for each partition and dataset. Existing processed files are reused automatically; if an input CSV changes, remove its corresponding processed file before regenerating the graphs.


### 4. Train scoring models using graph data

The supported architectures are 'GIN', 'GCN', 'GAT', 'GraphSAGE', and 'GraphTransformer'. For architecture comparisons, use the same CSV partitions and label definitions. Use a separate output directory for every architecture, hyperparameter configuration, and seed.

The trainer uses the training partition to update model parameters. It uses the validation partition to select `best.pth` and never reads the test partition. The default selection criteria are validation accuracy for classification and validation R² for regression; validation MCC or MAE can be selected explicitly. Use the criterion actually used for the corresponding paper experiment.

Outputs are:

- `best.pth`: the validation-selected model state dictionary.
- `config.json`: architecture, optimization, label-threshold, partition, and seed settings.
- `best_validation.json`: selected epoch and validation metrics.
- `training_log.csv`: training loss and validation metrics by epoch.
 

### 5. Reinforcement learning and molecular generation

The integrated reinforcement-learning workflow is implemented in `Trainer.py`. During training, PepBrick samples linear peptidomimetics, applies the cyclization/linker policy, evaluates the resulting structures using pretrained scoring models, and updates the generation policies.

The current example combines cyclic-peptide permeability prediction, MOR docking-score prediction, and neuropeptide classification. Molecules are generated during reinforcement learning.

#### Required inputs

Prepare the following resources before running the trainer:

| Resource | Description |
|---|---|
| `Data/Dataset/SMILES/` | SMILES tokenizer and vocabulary |
| `Data/Model/Final.pth` | Initial SMILES-generator weights loaded by the linker generators; the amino-acid generator is initialized within `RLWrapper` |
| `AA_contribution.pth` | Pretrained residue-contribution model, loaded from the repository root |
| Permeability scoring checkpoint | Loaded by `ScoringFunction(name="Cyclic_reg", func="regression")` |
| MOR scoring checkpoint | Loaded by `ScoringFunction(name="Opioid", func="regression")` |
| Neuropeptide scoring checkpoint | Loaded by `ScoringFunction(name="Neuro", func="classification")` |

#### Run training and generation

Run the following command from the repository root:

```bash
python Trainer.py
```

The default execution block is:

```python
if __name__ == "__main__":
    trainer = TrainerRL(
        max_len=6,
        save_dir="Data/Result/generation/new",
    )
    trainer.train(
        iteration=100,
        batch_size=32,
        save_every=1,
        debug_anomaly=False,
    )
```

Modify these arguments to configure the run:

| Argument | Description |
|---|---|
| `max_len` | Number of residue-generation decisions per peptidomimetic |
| `save_dir` | Directory for generated-sample CSV files |
| `iteration` | Number of reinforcement-learning iterations |
| `batch_size` | Target number of sampled linear structures per iteration |

With within-iteration deduplication enabled, the trainer attempts to collect the requested batch size using canonical SMILES and a bounded number of resampling rounds.

#### Reward and sampling settings

The current example calculates the molecular reward as:

```python
reward = (
    0.7 * permeability_score
    + 0.3 * MOR_score
    + 0.1 * neuropeptide_score
    + nnaa_bonus
)
```

The property scores are the reward-transformed outputs returned by the scoring models. The NNAA bonus is 0.1 for linear molecules that contain a sampled noncanonical residue and pass the corresponding scoring-validity check. The reward difference between the structures before and after cyclization guides the cyclization policy.


| Setting | Default |
|---|---|
| Noncanonical amino-acid sampling pattern | `N[C@@H](C(=O)O)*` |
| Linker sampling pattern | `C*` |
| Amino-acid SMILES sampling temperature | 1.5 |
| Linker SMILES sampling temperature | 1.5 |
| Top-k for both SMILES generators | 20 |
| Amino-acid SMILES generation limit | 100 generation steps |
| Natural-residue entropy coefficient | 0.001 |
| Gate entropy coefficient | 0.01 |
| Noncanonical-residue entropy coefficient | 0.001 |
| Cyclization entropy coefficient | 0.1 |

These entropy terms encourage exploration during reinforcement learning. Different optimization objectives require corresponding changes to the scoring models and reward expression.

#### Generated outputs

At the specified export interval, the trainer writes:

```text
Data/Result/generation/new/step_<iteration>.csv
```

Each file contains the structures sampled during that iteration and their associated predictions and rewards:

| Column | Description |
|---|---|
| `SMILES` | Linear peptidomimetic structure |
| `SMILES_cyc` | Structure returned after applying the cyclization/linker policy |
| `SMILES_linker` | Sampled linker structure, where applicable |
| `Has_NNAA` | Whether a noncanonical residue was sampled |
| `Is_Valid` | Validity mask used by the training procedure |
| `cyc_type` | Selected cyclization action |
| `True_permea_Before`, `True_permea_After` | Predicted permeability before and after cyclization |
| `True_affi_Before`, `True_affi_After` | Predicted MOR docking scores before and after cyclization |
| `True_neur_Before`, `True_neur_After` | Predicted neuropeptide classes before and after cyclization |
| `Total_Reward` | Reward after cyclization |
| `Total_Reward_delta` | Reward after cyclization minus reward before cyclization |
| `Position_<i>` | Residue identity or NNAA indicator at position `<i>` |
| `NNAA_<i>` | Noncanonical-residue SMILES at position `<i>`, where applicable |
| `Contribution_<i>` | Predicted residue contribution at position `<i>` |

The console reports policy, subgenerator, critic, and linker losses, together with mean rewards, validity rates, and NNAA usage. The current trainer exports sampling results during optimization.
