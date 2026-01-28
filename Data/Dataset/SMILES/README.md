---
license: bigscience-openrail-m
widget:
- text: O=C([C@@H](c1ccc(cc1)O)N)[MASK][C@@H]1C(=O)N2[C@@H]1SC([C@@H]2C(=O)O)(C)C
datasets:
- ChEMBL
pipeline_tag: fill-mask
tags:
- biology
- medical
---

# BERT base for SMILES
This is bidirectional transformer pretrained on SMILES (simplified molecular-input line-entry system) strings. 

Example: Amoxicillin
```
O=C([C@@H](c1ccc(cc1)O)N)N[C@@H]1C(=O)N2[C@@H]1SC([C@@H]2C(=O)O)(C)C
```

Two training objectives were used: 
1. masked language modeling
2. molecular-formula validity prediction

## Intended uses
This model is primarily aimed at being fine-tuned on the following tasks:
- molecule classification
- molecule-to-gene-expression mapping
- cell targeting

## How to use in your code
```python
from transformers import BertTokenizerFast, BertModel
checkpoint = 'unikei/bert-base-smiles'
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
model = BertModel.from_pretrained(checkpoint)

example = 'O=C([C@@H](c1ccc(cc1)O)N)N[C@@H]1C(=O)N2[C@@H]1SC([C@@H]2C(=O)O)(C)C'
tokens = tokenizer(example, return_tensors='pt')
predictions = model(**tokens)
```