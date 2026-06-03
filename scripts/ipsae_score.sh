#!/usr/bin/env bash
# Curated by Recep Adiyaman — https://github.com/recep2244/openfold3-finetune-kit
# =============================================================================
# ipsae_score.sh — interface confidence (ipSAE) for a predicted complex.
# -----------------------------------------------------------------------------
# ipSAE (Dunbrack et al., 2025) ranks protein-protein/-ligand interfaces from a
# model's PAE — a NO-reference metric (needs only the prediction, no experimental
# structure), complementing evaluate.sh's reference-based scoring. Useful for
# triaging fine-tuned predictions on targets you cannot yet validate.
#
# Good interface: ipSAE_min > 0.61 (standard) / > 0.70 (stringent).
#
# USAGE:
#   bash scripts/ipsae_score.sh <pae.(json|npz)> <model.(cif|pdb)> [pae_cut] [dist_cut]
#   # OpenFold3 / AF3-style outputs: pass the *_full_data*.json and the .cif, e.g.
#   bash scripts/ipsae_score.sh out/target/*full_data_0.json out/target/*_model.cif 10 10
# =============================================================================
set -euo pipefail

PAE="${1:?usage: ipsae_score.sh <pae.json|.npz> <model.cif|.pdb> [pae_cut=10] [dist_cut=10]}"
MODEL="${2:?missing model structure (.cif/.pdb)}"
PAE_CUT="${3:-10}"
DIST_CUT="${4:-10}"
IPSAE_DIR="${IPSAE_DIR:-$HOME/.cache/IPSAE}"

command -v python >/dev/null 2>&1 || { echo "python not found"; exit 1; }
if [[ ! -f "$IPSAE_DIR/ipsae.py" ]]; then
  echo "Fetching IPSAE into $IPSAE_DIR ..."
  git clone --depth 1 https://github.com/DunbrackLab/IPSAE.git "$IPSAE_DIR" || {
    echo "Could not clone IPSAE. Clone it manually: git clone https://github.com/DunbrackLab/IPSAE.git"; exit 1; }
fi

echo "ipSAE: $PAE + $MODEL  (pae_cut=$PAE_CUT dist_cut=$DIST_CUT)"
python "$IPSAE_DIR/ipsae.py" "$PAE" "$MODEL" "$PAE_CUT" "$DIST_CUT"
echo "Interpretation: ipSAE_min > 0.61 indicates a confident interface (> 0.70 stringent)."
