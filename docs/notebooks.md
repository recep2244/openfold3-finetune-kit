# Notebooks (cloud)

Notebook equivalents of the pipeline, for running on hosted GPU platforms without a terminal.
Both forms are maintained — pick whichever fits your workflow. Each notebook mirrors a script
in `scripts/`.

| Notebook | What it does | Script equivalent |
|---|---|---|
| `01_setup_and_verify.ipynb` | Install OpenFold3 + weights, run a smoke prediction | `verify_setup.sh` |
| `02_finetune_pipeline.ipynb` | Full pipeline: prepare → check → fine-tune → evaluate | `run_all.sh` |
| `03_inference.ipynb` | Predict a structure from a sequence and view it in 3D | `run_openfold predict` |

## Open in Colab

- **Setup & verify** — [open](https://colab.research.google.com/github/recep2244/openfold3-finetune-kit/blob/main/notebooks/01_setup_and_verify.ipynb)
- **Fine-tune pipeline** — [open](https://colab.research.google.com/github/recep2244/openfold3-finetune-kit/blob/main/notebooks/02_finetune_pipeline.ipynb)
- **Inference** — [open](https://colab.research.google.com/github/recep2244/openfold3-finetune-kit/blob/main/notebooks/03_inference.ipynb)

## Other platforms

The notebooks are platform-agnostic; on any hosted GPU Jupyter they work the same way:

- **Kaggle** — *New Notebook → File → Import Notebook → GitHub*, paste the notebook URL, then enable a GPU accelerator in the sidebar.
- **Paperspace Gradient / Lightning AI** — create a notebook on a GPU instance and upload the `.ipynb`, or clone the repo in a terminal cell.
- **SageMaker Studio Lab** — clone the repo and open the notebook; choose the GPU runtime.

## How they work

1. **Install** OpenFold3 via pip (`pip install "openfold3[cuequivariance]"`) — the simplest path on hosted platforms (a local workstation is better served by the pixi setup in [Getting started](getting-started.md)).
2. **Weights** are downloaded by `setup_openfold`, which is interactive — the notebooks feed its prompts non-interactively.
3. **Run** the relevant stage. The fine-tune notebook exposes `TRAIN_IDS` / `VAL_IDS` / `GPU_PROFILE` in a parameters cell.

!!! warning "GPU size"
    A free **T4 (16 GB)** runs the **smoke profile only** (`GPU_PROFILE="small"`). A real
    fine-tune needs an **A100/H100**. Inference (notebook 03) is light and runs on a T4.

!!! tip "Persist your results"
    Hosted disks are ephemeral. The fine-tune notebook saves to Google Drive on Colab; on other
    platforms, download `eval/out/results.csv` and the checkpoint before the session ends.

See [Getting started](getting-started.md) and the [Tutorial](tutorial.md) for the underlying steps.
