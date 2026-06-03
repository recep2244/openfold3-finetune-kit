# Tutorial: fine-tune on a target

A complete worked run, from a verified install to a baseline-vs-fine-tuned comparison. The
example uses the **PDE10A** set from the published low-N case study; substitute your own
target by changing two ID lists.

!!! note "Prerequisites"
    Complete [Getting started](getting-started.md) first — you need a working `run_openfold`,
    the `of3-p2-155k.pt` weights, and a green `verify_setup.sh`.

## 1. Choose your structures

You need two disjoint sets of 4-character PDB codes for the **same target**:

- **Train** (~10): complexes the model learns from.
- **Held-out** (~10–17): complexes the model must never see in training; you score on these.

Good held-out picks are recent structures (likely post-dating the base model's training) that
the baseline currently gets wrong — that is where fine-tuning has the most room to help. The
defaults in `examples/pde10a_target.txt` are a sound first run.

## 2. Configure the run

Edit the CONFIG block at the top of `scripts/run_all.sh`:

```bash
OF3_REPO="$HOME/openfold-3"                          # your openfold-3 clone
WORK="$HOME/openfold3_run"                            # results directory
CKPT="$HOME/.openfold3/of3-p2-155k.pt"                # base weights
TRAIN_IDS="5SDY 5SIQ 5SI7 5SIG 5SI5 5SI8 5SIY 5SG5 5SGL 5SIH"
VAL_IDS="5SH0 5SE0 5SHR 5SJL 5SH8 5SF4 5SFG 5SE5 5SHK 5SEE"
GPU_PROFILE="big"                                     # "big" (>=24 GB) or "small" (12 GB test)
```

## 3. Run the pipeline

```bash
bash scripts/run_all.sh
```

This checks readiness, prepares data (structures + ColabFold MSAs), runs the preflight,
fine-tunes, and evaluates baseline vs fine-tuned — see [How it works](pipeline.md) for what
each stage does. It is largely unattended; the fine-tune is the long part.

!!! tip "Step-by-step alternative"
    `run_all.sh` simply chains the stages. To run them individually, see
    [Usage reference](usage.md#step-by-step-instead-of-run_allsh).

## 4. Read the result

The run prints a comparison table:

```text
metric        baseline  finetuned   delta
lddt            0.71       0.74     +0.03
dockq_ave       0.41       0.58     +0.17
lddt_pli        0.55       0.72     +0.17
ligand_rmsd     3.20       1.10     -2.10
```

Higher is better for `lddt`, `dockq_ave`, and `lddt_pli`; lower is better for `ligand_rmsd`.
If interface metrics rise and ligand RMSD falls while `lddt` holds, the fine-tune worked.
Per-structure numbers are in `<WORK>/eval/out/results.csv`.

## 5. Confirm you did not regress elsewhere

Run a couple of unrelated targets through prediction with both checkpoints (the "forgetting
check") to confirm the model did not get worse at everything else.

## Limited GPU?

A full fine-tune does not fit on a 12–16 GB GPU. Run the smoke test to validate the pipeline
end-to-end on tiny data — see [Small GPU (12 GB)](small-gpu.md) — and move to an 80 GB
instance for the real run ([Cloud GPU](cloud.md)).

## Day-to-day loop

```bash
cd ~/openfold-3 && pixi shell -e openfold3-cuda12   # activate the environment
cd ~/openfold3-finetune-kit                          # the kit
# edit TRAIN_IDS / VAL_IDS for a new target
bash scripts/run_all.sh
```

If anything fails, see [Troubleshooting](troubleshooting.md).
