---
license: apache-2.0
base_model: openfold3-p2-155k
tags:
  - protein-structure
  - openfold3
  - alphafold3
  - co-folding
  - fine-tuned
library_name: openfold3
---

# OpenFold3 fine-tuned on <TARGET>

Fine-tuned [OpenFold3](https://github.com/aqlaboratory/openfold-3) weights,
adapted to a single target family from the public `openfold3-p2-155k` checkpoint
using [openfold3-finetune-kit](https://github.com/recep2244/openfold3-finetune-kit).

## Intended use
Co-folding (joint protein + small-molecule) structure prediction for **<TARGET>**.
Produces a `.ckpt` consumable by `run_openfold predict --inference-ckpt-path`.

## Training
- **Base:** `openfold3-p2-155k`
- **Data:** ~10 protein–ligand complexes of the target (held-out set kept separate)
- **Recipe:** low-n fine-tune — LR 3e-4, warmup 50 steps, EMA 0.99, templates off,
  interface-biased crops (`configs/finetune_lowN_single_gpu.yml`)
- **Steps / hardware:** ~350 steps, ~20 h on one 80 GB GPU

## Evaluation (baseline → fine-tuned, held-out set)
| metric | baseline | finetuned |
|---|---|---|
| interface lDDT | _fill in_ | _fill in_ |
| DockQ (avg) | _fill in_ | _fill in_ |
| lDDT-PLI | _fill in_ | _fill in_ |
| ligand RMSD (Å) | _fill in_ | _fill in_ |

## Limitations
- Tuned for one target; may **forget** unrelated targets — run a forgetting check.
- Inherits OpenFold3's limitations on novel chemistry far from training data.

## License
Apache-2.0, inherited from the base OpenFold3 weights. Keep attribution to the
OpenFold Consortium / AlQuraishi Lab.
