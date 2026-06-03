"""
Gradio Space: predict a protein (optionally + ligand) structure with OpenFold3.

Runs the OpenFold3 CLI under the hood (templates OFF, MSAs from the ColabFold
server) and renders the predicted structure with 3Dmol.js. Point INFERENCE_CKPT
at a fine-tuned checkpoint to serve your own tuned model.

Needs a GPU Space. Install deps from requirements.txt (includes openfold3).

Curated by Recep Adiyaman — https://github.com/recep2244/openfold3-finetune-kit
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

# Use a fine-tuned checkpoint if provided, else the default downloaded weights.
INFERENCE_CKPT: str = os.environ.get("INFERENCE_CKPT", "").strip()

# Generous ceiling so the Space doesn't hang on an accidental whole-proteome paste.
MAX_RESIDUES = 1200
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

VIEWER_TEMPLATE = """
<div id="viewer" role="img" aria-label="Predicted 3D structure"
     style="width:100%;height:480px;position:relative;border-radius:10px;overflow:hidden;"></div>
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
    chains: list[dict] = [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": sequence}]
    if smiles:
        chains.append({"molecule_type": "ligand", "chain_ids": ["L"], "smiles": smiles})
    return {"queries": {"prediction": {"chains": chains}}}


def _validate(sequence: str) -> str:
    seq = "".join((sequence or "").split()).upper()
    if not seq:
        raise gr.Error("Please enter a protein sequence.")
    invalid = sorted(set(seq) - VALID_AA)
    if invalid:
        raise gr.Error(f"Sequence contains non-amino-acid characters: {', '.join(invalid)}")
    if len(seq) > MAX_RESIDUES:
        raise gr.Error(f"Sequence is {len(seq)} residues; this demo caps at {MAX_RESIDUES}.")
    return seq


def predict(sequence: str, smiles: str, num_samples: int) -> tuple[str, str]:
    """Predict a structure and return (3D viewer HTML, path to the CIF file)."""
    seq = _validate(sequence)
    smiles = (smiles or "").strip()

    workdir = Path(tempfile.mkdtemp(prefix="of3_"))
    query_path = workdir / "query.json"
    query_path.write_text(json.dumps(_build_query(seq, smiles)))

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

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError as exc:
        raise gr.Error(
            "run_openfold is not installed in this Space (needs the openfold3 package)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise gr.Error(
            "Prediction timed out (30 min). Try a shorter sequence or fewer samples."
        ) from exc

    # run_openfold can exit 0 even when a query fails — verify a structure exists.
    cifs = sorted((workdir / "out").rglob("*.cif"))
    if not cifs:
        tail = (proc.stderr or proc.stdout or "no output")[-1500:]
        raise gr.Error(f"Prediction produced no structure.\n\n{tail}")

    cif_text = cifs[0].read_text()
    viewer = VIEWER_TEMPLATE.format(cif_json=json.dumps(cif_text))
    return viewer, str(cifs[0])


with gr.Blocks(
    title="OpenFold3 structure prediction",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="teal"),
) as demo:
    gr.Markdown(
        "# OpenFold3 structure prediction\n"
        "Predict a protein — optionally with a small-molecule ligand — and view it in 3D. "
        "Templates are off; MSAs come from the ColabFold server. **A GPU runtime is required.**"
    )
    with gr.Row():
        with gr.Column(scale=1):
            seq = gr.Textbox(
                label="Protein sequence (one-letter)",
                info="Standard 20 amino acids; whitespace is ignored.",
                lines=5,
                placeholder=UBIQUITIN,
            )
            lig = gr.Textbox(
                label="Ligand SMILES (optional)",
                info="Leave blank for protein-only prediction.",
                placeholder="CC(=O)Oc1ccccc1C(=O)O",
            )
            ns = gr.Slider(
                1,
                5,
                value=1,
                step=1,
                label="Diffusion samples",
                info="More samples = more diversity, slower.",
            )
            btn = gr.Button("Predict structure", variant="primary")
        with gr.Column(scale=1):
            out_view = gr.HTML(label="Predicted structure")
            out_file = gr.File(label="Download CIF")

    gr.Examples(
        examples=[[UBIQUITIN, "", 1], [UBIQUITIN, "CC(=O)Oc1ccccc1C(=O)O", 1]],
        inputs=[seq, lig, ns],
        label="Examples",
    )
    gr.Markdown(
        "Part of [openfold3-finetune-kit](https://github.com/recep2244/openfold3-finetune-kit) "
        "· [Documentation](https://recep2244.github.io/openfold3-finetune-kit/docs/) "
        "· Curated by [Recep Adiyaman](https://recep2244.github.io/portfolio/)"
    )

    btn.click(predict, inputs=[seq, lig, ns], outputs=[out_view, out_file], api_name="predict")

if __name__ == "__main__":
    demo.launch()
