# 🧪 PepBrick:  Design of Peptidomimetics Using a Multi-level Nested Structured Reinforcement Learning Model

---

## 📄 Abstract
> **Abstract**  
Peptide molecules, owing to their vast chemical space and considerable modification potential, enable the balanced optimization of multiple properties in specific therapeutic scenarios and the development of ideal drug candidates. Thus, developing artificial intelligence approaches to overcome the cost inefficiencies of traditional trial-and-error methods and design multi-objective optimized peptidomimetics with extensive chemical space exploration has become an imperative challenge. To this end, we developed PepBrick, a structured multi-level nested reinforcement learning framework that explores the chemical-bioactivity space landscape of peptidomimetics via a hierarchical residue-sequence-topology progression strategy. Through a series of systematic experiments on isolated modules and proof-of-concept applications in the design of BBB-penetrating neuropeptides and tumor fluorescent probes, we fully validated the framework’s potential and robustness, which establishes a versatile and innovative paradigm for the development of next-generation peptide-based therapeutics.

---

## 🖼️ Overview
> **The PepBrick framework：** (i) Sequence Generation Module: Selects canonical amino acids from the candidate pool or activates the unnatural amino acid generation module when needed. (iii) Amino Acid Generation Module: Activated by the sequence generation module to produce diverse unnatural amino acid structures. (ii) Cyclization Policy Module: Determines the cyclization strategy and sites for linear peptidomimetics and decides whether to activate the linker generator. (iv) Linker Generation Module: Activated by the cyclization policy module to generate structurally feasible linker molecules. (v) Scoring Module: Incorporates a global-level graph neural network encoder, an amino-acid-level language model, and a Critic network to enhance training stability. This module outputs the reward signal for generated peptidomimetics and updates the parameters of the entire PepBrick framework.}

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
