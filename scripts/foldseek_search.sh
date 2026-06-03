#!/usr/bin/env bash
# Curated by Recep Adiyaman — https://github.com/recep2244/openfold3-finetune-kit
# =============================================================================
# foldseek_search.sh — structural homolog / novelty search for a target.
# -----------------------------------------------------------------------------
# Two uses in a fine-tuning workflow:
#   * Curating sets   — find structural homologs of your target to assemble
#                       train / held-out structures.
#   * Novelty check   — gauge how far a target/pose sits from known structures
#                       (the regime where the base model is weakest).
#
# Needs Foldseek:  conda install -c conda-forge -c bioconda foldseek
# Set up a database once, e.g.:  foldseek databases PDB pdb100 tmp/
#
# USAGE:
#   bash scripts/foldseek_search.sh <query.(pdb|cif)> [db=pdb100] [out.m8]
# =============================================================================
set -euo pipefail

QUERY="${1:?usage: foldseek_search.sh <query.pdb|.cif> [db=pdb100] [out.m8]}"
DB="${2:-pdb100}"
OUT="${3:-foldseek_results.m8}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

command -v foldseek >/dev/null 2>&1 || {
  echo "Foldseek not installed. Install: conda install -c conda-forge -c bioconda foldseek"; exit 1; }
[[ -f "$QUERY" ]] || { echo "query not found: $QUERY"; exit 1; }

echo "Foldseek easy-search: $QUERY vs $DB"
foldseek easy-search "$QUERY" "$DB" "$OUT" "$TMP" \
  --format-output "query,target,pident,alnlen,evalue,bits" >/dev/null

hits=$(wc -l < "$OUT" | tr -d ' ')
echo "hits: $hits   (written to $OUT)"
if [[ "$hits" -gt 0 ]]; then
  echo "top hits (query target %id alnlen evalue bits):"
  sort -k6 -gr "$OUT" | head -5 | sed 's/^/  /'
  top=$(sort -k6 -gr "$OUT" | head -1 | awk '{print $3}')
  awk -v t="$top" 'BEGIN{ if (t+0 < 30) print "novelty: NOVEL fold (top identity " t "% < 30%)";
                          else print "novelty: known-like (top identity " t "%)" }'
else
  echo "no hits — try a larger database (afdb50) or a higher -e."
fi
