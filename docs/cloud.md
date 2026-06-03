# Running a real fine-tune on a rented 80 GB GPU

A full fine-tune does not fit on a small (12–16 GB) GPU. The proven recipe wants
an **80 GB H100 or A100-80GB**. Renting one by the hour is the cheapest path.

## 1. Rent the box
Any provider works (RunPod, Lambda, Vast.ai, CoreWeave). Pick a **Linux image
with NVIDIA driver + CUDA 12** preinstalled, single **H100 80GB** or **A100 80GB**.

Budget: the PDE10A recipe is ~350 steps ≈ **~20 h on one H100**. Baseline and
evaluation passes are cheap (evaluation runs 5 samples per held-out structure).

## 2. Install (same as local)
```bash
curl -fsSL https://pixi.sh/install.sh | sh && exec "$SHELL"
pixi self-update                                   # need pixi >= 0.68
git clone https://github.com/aqlaboratory/openfold-3.git ~/openfold-3
( cd ~/openfold-3 && pixi run -e openfold3-cuda12 setup_openfold )   # env + ~2 GB weights
docker pull registry.scicore.unibas.ch/schwede/openstructure:latest # for eval scoring
```
The pixi env provides its own matched nvcc + cutlass — **do not** set a system
`CUDA_HOME`, or the evoformer kernel build will fail.

## 3. Copy this kit and run
```bash
scp -r openfold3-finetune-kit/  user@cloudbox:~/openfold3-finetune-kit
# on the box:
cd ~/openfold3-finetune-kit && chmod +x scripts/*.sh
bash scripts/verify_setup.sh
# edit scripts/run_all.sh: set GPU_PROFILE="big" and your TRAIN_IDS / VAL_IDS
( cd ~/openfold-3 && pixi shell -e openfold3-cuda12 )   # activate, then:
bash scripts/run_all.sh
```

## 4. Save results before destroying the instance
Cloud disks vanish when the instance dies. Pull these back first:
```bash
scp -r user@cloudbox:~/openfold3_run/train_out/checkpoints  ./checkpoints
scp     user@cloudbox:~/openfold3_run/eval/out/results.csv  ./results.csv
```

## Scaling past your own data
For chemistry your own structures don't cover, the field uses **federated
fine-tuning** (exchange privacy-preserving updates, never raw structures).
See [Background & rationale](background.md) for context.
