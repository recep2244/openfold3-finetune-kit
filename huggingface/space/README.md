---
title: OpenFold3 Structure Prediction
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: t4-medium
---

# OpenFold3 Structure Prediction

Predict a protein (optionally + ligand) structure with [OpenFold3](https://github.com/aqlaboratory/openfold-3).
Templates are off; MSAs are fetched from the ColabFold server.

**This Space needs a GPU.** On CPU hardware it will fail to run the model.

To serve a **fine-tuned** checkpoint instead of the default weights, set the
Space secret/variable `INFERENCE_CKPT` to the path of your `.ckpt`/`.pt` file
(e.g. a file downloaded from your Hugging Face model repo at startup).

Part of [openfold3-finetune-kit](https://github.com/recep2244/openfold3-finetune-kit).
