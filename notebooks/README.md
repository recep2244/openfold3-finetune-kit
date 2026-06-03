# Notebooks

Notebook equivalents of the shell pipeline, for running on hosted GPU platforms
(Google Colab, Kaggle, Paperspace Gradient, SageMaker Studio Lab, Lightning AI) without
touching a terminal. Both forms are maintained — use whichever fits your workflow.

| Notebook | Does | Script equivalent |
|---|---|---|
| [`01_setup_and_verify.ipynb`](01_setup_and_verify.ipynb) | Install OpenFold3 + weights, run a smoke prediction | `scripts/verify_setup.sh` |
| [`02_finetune_pipeline.ipynb`](02_finetune_pipeline.ipynb) | Full pipeline: prepare → check → fine-tune → evaluate | `scripts/run_all.sh` |
| [`03_inference.ipynb`](03_inference.ipynb) | Predict a structure from a sequence and view it in 3D | `run_openfold predict` |

## Open in Colab
- Setup & verify — https://colab.research.google.com/github/recep2244/openfold3-finetune-kit/blob/main/notebooks/01_setup_and_verify.ipynb
- Fine-tune pipeline — https://colab.research.google.com/github/recep2244/openfold3-finetune-kit/blob/main/notebooks/02_finetune_pipeline.ipynb
- Inference — https://colab.research.google.com/github/recep2244/openfold3-finetune-kit/blob/main/notebooks/03_inference.ipynb

On Kaggle/Paperspace/others, open the `.ipynb` directly and select a GPU runtime.

## Notes
- The notebooks use the **pip** install path (simplest on hosted platforms). A local
  workstation is better served by the pixi setup in [Getting started](../docs/getting-started.md).
- `setup_openfold` is interactive; the notebooks feed its prompts non-interactively.
- A free **T4 (16 GB)** runs the smoke profile only; a real fine-tune needs an **A100/H100**.
- Cloud disks are ephemeral — persist `\$WORK/eval/out/results.csv` and the checkpoint before the session ends.
