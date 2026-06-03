# Getting started

This page takes you from nothing to a verified OpenFold3 install. For a full worked
fine-tune, continue to the [Tutorial](tutorial.md).

## Prerequisites

- Linux with an NVIDIA GPU and driver supporting **CUDA ≥ 12.1**.
- ~10 GB free disk for the environment and model weights.
- **For data preparation (`run_all.sh`):** two tools the OpenFold3 scripts need that the
  `openfold3-cuda12` env does not always include — install them into the env:
  ```bash
  pip install biopython                                 # MSA representatives step
  conda install -c conda-forge -c bioconda mmseqs2      # sequence clustering in the dataset cache
  ```
- For evaluation scoring only: Docker (to run the OpenStructure image).

!!! note "OpenFold3 is a moving target — pin a commit"
    This kit tracks `aqlaboratory/openfold-3` **`main`**, which changes its data-prep schema
    over time (e.g. the MSA key `bfd_uniref_hits` → `bfd_hits`). For reproducibility, check out
    a known-good commit (`git -C ~/openfold-3 checkout <sha>`) and record it in your
    [reference run](https://recep2244.github.io/openfold3-finetune-kit/). The kit's scripts
    target current `main` as of mid-2026.

## 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi self-update          # the openfold-3 workspace requires pixi >= 0.68
```

!!! warning "pixi version"
    A fresh install can be older than 0.68, in which case `setup_openfold` exits
    immediately with *"workspace requires pixi '>=0.68'"*. Always run `pixi self-update`.

## 2. Install OpenFold3 and download weights

```bash
git clone https://github.com/aqlaboratory/openfold-3.git ~/openfold-3
cd ~/openfold-3
pixi run -e openfold3-cuda12 setup_openfold
```

`setup_openfold` is interactive (cache directory, which checkpoint, integration tests).
Accept the defaults, choose the default checkpoint, and skip the integration tests; it
downloads `of3-p2-155k.pt` (~2.2 GB) to `~/.openfold3/`.

!!! danger "Do not set `CUDA_HOME` under pixi"
    The pixi environment ships its own matched `nvcc` and cutlass. Exporting a *system*
    `CUDA_HOME` over it makes the DeepSpeed evoformer kernel fail to compile. Leave
    `CUDA_HOME` unset. (The `CUDA_HOME`/`CUTLASS_PATH` advice you see elsewhere applies
    only to pip/conda installs — see [Troubleshooting](troubleshooting.md).)

## 3. (Optional) Docker image for evaluation scoring

```bash
docker pull registry.scicore.unibas.ch/schwede/openstructure:latest
```

Without it, training and prediction still work; only the scoring step is skipped.

## 4. Verify

```bash
git clone https://github.com/recep2244/openfold3-finetune-kit.git
cd openfold3-finetune-kit
make verify        # or: bash scripts/verify_setup.sh
```

A healthy result ends with:

```text
SUMMARY:  10 passed, 1 warnings, 0 failed
```

The single warning is expected on GPUs under 24 GB (smoke-test profile only). You are
now ready for the [Tutorial](tutorial.md).
