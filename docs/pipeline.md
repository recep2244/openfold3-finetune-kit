# How it works

`scripts/run_all.sh` orchestrates four stages and stops with a plain-English message if
any of them fails.

```mermaid
flowchart TD
    A[run_all.sh] --> B[A. Readiness check<br/>tools, repo, weights]
    B --> C[B. Data prep<br/>prepare_data.sh]
    C --> D[B2. Preflight<br/>check_data.sh]
    D --> E[C. Fine-tune<br/>run_openfold train]
    E --> F[D. Evaluate<br/>evaluate.sh]
```

## A. Readiness check
Confirms `run_openfold` is on `PATH`, the OpenFold3 repo and weights exist, and selects the
fine-tune config from `GPU_PROFILE` (`big` → `configs/finetune_lowN_single_gpu.yml`,
`small` → `configs/finetune_test_12gb.yml`).

## B. Data preparation
`prepare_data.sh` downloads each PDB structure, preprocesses it into OpenFold3's `.npz`
format with reference molecules, and retrieves MSAs.

!!! info "MSAs only — no databases, no templates"
    Alignments are fetched from the **ColabFold server** (`run_openfold align-msa-server`).
    Templates are disabled everywhere (`n_templates: 0`, `--use-templates False`), so the
    only external dependency is the MSA fetch. **Training itself never contacts the network** —
    it reads the alignment arrays saved during this stage.

## B2. Preflight
`check_data.sh` verifies every required artifact exists and is non-empty — preprocessed
structures, reference molecules, alignment arrays, and dataset caches — and guards known
OpenFold3 input pitfalls (dots in query names, chain-level MSA keys, alignment formats)
*before* any GPU time is spent.

## C. Fine-tune
`run_all.sh` renders a runner YAML from the chosen config with your data paths, then calls
`run_openfold train`. The recipe loads the public `of3-p2-155k.pt` weights directly
(`manual_checkpoint_loading: true`) and applies the low-N overrides described in
[Configuration](configuration.md). Checkpoints are written under `<work>/train_out/checkpoints/`.

## D. Evaluation
`evaluate.sh` predicts each held-out structure with both the **baseline** and the
**fine-tuned** checkpoint (5 diffusion samples each), scores them against the experimental
structures with OpenStructure, and prints a comparison table. Interface lDDT, DockQ, and
lDDT-PLI should rise and ligand RMSD should fall while global lDDT holds steady.

## Forgetting check
A fine-tune adapts to one target and may regress on unrelated ones. After a run, predict a
few unrelated targets with both checkpoints to confirm general performance is preserved.
