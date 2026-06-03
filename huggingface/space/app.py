"""
Gradio Space: predict a protein (optionally + ligand) structure with OpenFold3.

Runs the OpenFold3 CLI under the hood (templates OFF, MSAs from the ColabFold
server) and renders the predicted structure with 3Dmol.js. Point INFERENCE_CKPT
at a fine-tuned checkpoint to serve your own tuned model.

Needs a GPU Space. Install deps from requirements.txt (includes openfold3).

Curated by Recep Adiyaman — https://github.com/recep2244/openfold3-finetune-kit
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

# Use a fine-tuned checkpoint if provided, else the default downloaded weights.
INFERENCE_CKPT = os.environ.get("INFERENCE_CKPT", "").strip()

VIEWER_TEMPLATE = """
<div id="viewer" style="width:100%;height:480px;position:relative;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  const cif = {cif_json};
  const v = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
  v.addModel(cif, "cif");
  v.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
  v.addStyle({{hetflag: true}}, {{stick: {{}}}});
  v.zoomTo();
  v.render();
</script>
"""


def _build_query(sequence: str, smiles: str) -> dict:
    chains = [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": sequence.strip()}]
    if smiles.strip():
        chains.append({"molecule_type": "ligand", "chain_ids": ["L"], "smiles": smiles.strip()})
    return {"queries": {"prediction": {"chains": chains}}}


def predict(sequence: str, smiles: str, num_samples: int):
    sequence = (sequence or "").strip().upper()
    if not sequence or not sequence.isalpha():
        raise gr.Error("Enter a valid one-letter amino-acid sequence.")

    workdir = Path(tempfile.mkdtemp(prefix="of3_"))
    query_path = workdir / "query.json"
    query_path.write_text(json.dumps(_build_query(sequence, smiles)))

    cmd = [
        "run_openfold",
        "predict",
        "--query-json",
        str(query_path),
        "--use-msa-server",
        "True",
        "--use-templates",
        "False",
        "--num-diffusion-samples",
        str(int(num_samples)),
        "--num-model-seeds",
        "1",
        "--output-dir",
        str(workdir / "out"),
    ]
    if INFERENCE_CKPT:
        cmd += ["--inference-ckpt-path", INFERENCE_CKPT]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    # run_openfold can exit 0 even when a query fails — verify a structure exists.
    cifs = sorted((workdir / "out").rglob("*.cif"))
    if not cifs:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise gr.Error(f"Prediction produced no structure.\n\n{tail}")

    cif_text = cifs[0].read_text()
    viewer = VIEWER_TEMPLATE.format(cif_json=json.dumps(cif_text))
    return viewer, str(cifs[0])


with gr.Blocks(title="OpenFold3 structure prediction") as demo:
    gr.Markdown(
        "# OpenFold3 structure prediction\n"
        "Predict a protein (optionally with a small-molecule ligand) structure. "
        "Templates are off; MSAs come from the ColabFold server. **GPU required.**"
    )
    with gr.Row():
        with gr.Column():
            seq = gr.Textbox(
                label="Protein sequence (one-letter)",
                lines=4,
                placeholder="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            )
            lig = gr.Textbox(label="Ligand SMILES (optional)", placeholder="CC(=O)Oc1ccccc1C(=O)O")
            ns = gr.Slider(1, 5, value=1, step=1, label="Diffusion samples")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out_view = gr.HTML(label="Predicted structure")
            out_file = gr.File(label="Download CIF")
    btn.click(predict, inputs=[seq, lig, ns], outputs=[out_view, out_file])

if __name__ == "__main__":
    demo.launch()
