# Usage reference

All scripts live in `scripts/` and are driven by environment variables (or the CONFIG block
at the top of `run_all.sh`). Run them from the repo root.

## Scripts

| Script | Purpose |
|---|---|
| `verify_setup.sh` | Validate the install and run a real smoke prediction; prints PASS/FAIL |
| `run_all.sh` | End-to-end pipeline: prepare → check → train → evaluate |
| `prepare_data.sh` | Download structures and fetch ColabFold MSAs |
| `check_data.sh` | Preflight data-integrity check (run automatically by `run_all.sh`) |
| `evaluate.sh` | Score baseline vs fine-tuned on the held-out set |
| `run_small_test.sh` | Minimal end-to-end smoke test on tiny data |

## `run_all.sh` configuration

Edit the CONFIG block near the top of `scripts/run_all.sh`:

| Variable | Meaning |
|---|---|
| `OF3_REPO` | Path to the cloned `openfold-3` (default `~/openfold-3`) |
| `WORK` | Output directory for all results (default `~/openfold3_run`) |
| `CKPT` | Base weights (default `~/.openfold3/of3-p2-155k.pt`) |
| `TRAIN_IDS` | Space-separated PDB codes the model trains on |
| `VAL_IDS` | Space-separated held-out PDB codes (never seen in training) |
| `GPU_PROFILE` | `big` (≥24 GB) or `small` (12 GB smoke test) |

## Common environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `MSA_MODE` | `prepare_data.sh` | `colabfold` (default) or `snakemake` (local DBs, advanced) |
| `WORK` | most scripts | Shared results directory |
| `DEVICE` | `run_small_test.sh` | `cpu` forces a guaranteed-to-run CPU test |

## Make targets

| Target | Action |
|---|---|
| `make verify` | Run `verify_setup.sh` |
| `make lint` | shellcheck + yamllint + ruff + notebook validation |
| `make fmt` | Format Python with ruff |
| `make docs` | Serve the docs site locally |
| `make docs-build` | Build the docs site (`--strict`) |
| `make clean` | Remove local build/output artifacts |

## Notebooks (cloud platforms)

Notebook equivalents of the pipeline live in `notebooks/`, for running on hosted GPU
platforms (Colab, Kaggle, Paperspace, SageMaker Studio Lab, Lightning AI) without a terminal.

| Notebook | Script equivalent |
|---|---|
| `01_setup_and_verify.ipynb` | `verify_setup.sh` |
| `02_finetune_pipeline.ipynb` | `run_all.sh` |
| `03_inference.ipynb` | `run_openfold predict` |

They use the pip install path; a free T4 runs the smoke profile only. See `notebooks/README.md`.

## Step-by-step (instead of `run_all.sh`)

```bash
bash scripts/prepare_data.sh
run_openfold train --runner-yaml "$WORK/runner_finetune.yml" --seed 42
bash scripts/evaluate.sh
```
