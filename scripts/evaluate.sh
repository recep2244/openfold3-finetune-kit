#!/usr/bin/env bash
# =============================================================================
# evaluate.sh — baseline vs fine-tuned OpenFold3 on a held-out set.
# -----------------------------------------------------------------------------
# For every held-out query it:
#   1) runs N diffusion samples with the BASELINE (public) weights
#   2) runs N diffusion samples with your FINE-TUNED checkpoint
#   3) selects the highest-confidence sample (OF3's own 'sample_ranking_score')
#   4) scores both vs the experimental structure with OpenStructure:
#        - lDDT + DockQ + interface contact scores  (compare-structures)
#        - lDDT-PLI + ligand RMSD                    (compare-ligand-structures)
#   5) writes results.csv and prints baseline-vs-finetuned means
#
# Verified against openfold-3 source:
#   predict CLI flags        -> openfold3/run_openfold.py  (predict())
#   inference loads .ckpt EMA -> experiment_runner.py::setup + get_state_dict_from_checkpoint
#   output naming/ranking     -> openfold3/core/runners/writer.py
#       <out>/<qid>/seed_<seed>/<qid>_seed_<seed>_sample_<s>_model.cif
#       ..._confidences_aggregated.json  -> has "sample_ranking_score"
#
# KNOWN GOTCHAS folded in (from openfold-3 issues):
#   * #176 query IDs must NOT contain dots '.'  -> keep JSON query keys dot-free
#   * #149 DataLoader worker crashes            -> NWORKERS kept low/0-safe
#   * templates OFF for eval (matches PDE10A)   -> --use-templates False
#
# USAGE:  bash evaluate.sh
# =============================================================================
set -euo pipefail

# ----------------------------- CONFIG (edit me) ------------------------------
QUERY_DIR="${QUERY_DIR:-./eval/queries}"     # one query JSON per held-out target
REF_DIR="${REF_DIR:-./eval/references}"       # reference CIFs named <query_id>.cif
OUT="${OUT:-./eval/out}"

# Baseline weights: a registry name (downloaded automatically) OR an absolute .pt
BASELINE_CKPT_NAME="${BASELINE_CKPT_NAME:-openfold3-p2-155k}"
# Your fine-tuned Lightning checkpoint (loads EMA weights directly — no conversion):
FT_CKPT="${FT_CKPT:-./finetune_pde10a_output/checkpoints/last.ckpt}"

NSAMPLES="${NSAMPLES:-5}"                      # diffusion samples per target (PDE10A used 5)
USE_MSA_SERVER="${USE_MSA_SERVER:-True}"      # True -> ColabFold server; or precompute MSAs
LIGAND_SCORING="${LIGAND_SCORING:-true}"      # also run OST ligand scoring (protein-ligand)
OST_IMAGE="${OST_IMAGE:-registry.scicore.unibas.ch/schwede/openstructure:latest}"
# -----------------------------------------------------------------------------

mkdir -p "$OUT/baseline" "$OUT/finetuned" "$OUT/scores"
RESULTS="$OUT/results.csv"
echo "query_id,model,lddt,dockq_ave,ics,ips,lddt_pli,ligand_rmsd,best_sample,ranking_score" > "$RESULTS"

# --- helper: run inference for one model into one dir ---
run_model () {  # $1=query_json  $2=outdir  $3=ckpt_flag_array...
  local qjson="$1"; local odir="$2"; shift 2
  run_openfold predict \
    --query-json "$qjson" \
    --num-diffusion-samples "$NSAMPLES" \
    --use-msa-server "$USE_MSA_SERVER" \
    --use-templates False \
    --output-dir "$odir" \
    "$@"
}

# --- helper (python): pick best sample by sample_ranking_score ---
pick_best () {  # $1=model_root_for_query  -> prints "best_cif\tranking_score\tsample_n"
  python3 - "$1" <<'PY'
import sys, json, glob, os
root = sys.argv[1]
best = (None, -1e9, None)
for jf in glob.glob(os.path.join(root, "**", "*_confidences_aggregated.json"), recursive=True):
    try:
        d = json.load(open(jf))
    except Exception:
        continue
    # AF3 ranking score; fall back to avg_plddt if absent
    score = d.get("sample_ranking_score", d.get("avg_plddt", -1e9))
    cif = jf.replace("_confidences_aggregated.json", "_model.cif")
    if not os.path.exists(cif):
        cif = jf.replace("_confidences_aggregated.json", "_model.pdb")
    n = jf.split("_sample_")[-1].split("_")[0] if "_sample_" in jf else "?"
    if score > best[1]:
        best = (cif, float(score), n)
print(f"{best[0]}\t{best[1]}\t{best[2]}")
PY
}

# --- helper: OST scoring; echoes "lddt,dockq_ave,ics,ips,lddt_pli,ligand_rmsd" ---
score_ost () {  # $1=model_cif  $2=ref_cif
  local model="$1"; local ref="$2"
  local mdir; mdir="$(cd "$(dirname "$model")" && pwd)"
  local rdir; rdir="$(cd "$(dirname "$ref")" && pwd)"
  local mname; mname="$(basename "$model")"
  local rname; rname="$(basename "$ref")"
  local sj; sj="$OUT/scores/$(basename "$model" | tr '/.' '__')_cs.json"
  local lj; lj="$OUT/scores/$(basename "$model" | tr '/.' '__')_lig.json"

  # protein / interface metrics
  docker run --rm -v "$mdir:/m" -v "$rdir:/r" -v "$OUT/scores:/s" "$OST_IMAGE" \
    compare-structures -m "/m/$mname" -r "/r/$rname" \
    --fault-tolerant --min-pep-length 4 --min-nuc-length 4 \
    --lddt --dockq --ics --ips -o "/s/$(basename "$sj")" >/dev/null 2>&1 || true

  # ligand pose metrics (best-effort; protein-ligand only)
  if [[ "$LIGAND_SCORING" == "true" ]]; then
    docker run --rm -v "$mdir:/m" -v "$rdir:/r" -v "$OUT/scores:/s" "$OST_IMAGE" \
      compare-ligand-structures -m "/m/$mname" -r "/r/$rname" \
      --lddt-pli --rmsd -o "/s/$(basename "$lj")" >/dev/null 2>&1 || true
  fi

  python3 - "$sj" "$lj" <<'PY'
import sys, json, os
def g(path, *keys, default=""):
    if not os.path.exists(path): return default
    try: d = json.load(open(path))
    except Exception: return default
    for k in keys:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return default
    if isinstance(d, dict):
        # average over chain/interface entries if nested
        vals = [v for v in d.values() if isinstance(v, (int, float))]
        return round(sum(vals)/len(vals), 4) if vals else default
    return round(d, 4) if isinstance(d, (int, float)) else d
cs, lj = sys.argv[1], sys.argv[2]
lddt = g(cs, "lddt")
dockq = g(cs, "dockq_ave") or g(cs, "dockq")
ics  = g(cs, "ics")
ips  = g(cs, "ips")
pli  = g(lj, "lddt_pli")
lrms = g(lj, "rmsd") or g(lj, "bisy_rmsd")
print(f"{lddt},{dockq},{ics},{ips},{pli},{lrms}")
PY
}

echo "==> Evaluating: baseline=$BASELINE_CKPT_NAME  finetuned=$FT_CKPT  samples=$NSAMPLES"
shopt -s nullglob
for qjson in "$QUERY_DIR"/*.json; do
  # query IDs = top-level keys under "queries"
  mapfile -t QIDS < <(python3 -c "import json,sys;print('\n'.join(json.load(open('$qjson')).get('queries',{}).keys()))")

  echo "--- $(basename "$qjson")  (queries: ${QIDS[*]}) ---"
  run_model "$qjson" "$OUT/baseline"  --inference-ckpt-name "$BASELINE_CKPT_NAME"
  run_model "$qjson" "$OUT/finetuned" --inference-ckpt-path "$FT_CKPT"

  for qid in "${QIDS[@]}"; do
    ref="$REF_DIR/$qid.cif"
    [[ -f "$ref" ]] || { echo "    [skip] no reference $ref"; continue; }
    for model in baseline finetuned; do
      read -r bestcif rscore nsamp < <(pick_best "$OUT/$model/$qid")
      if [[ -z "${bestcif:-}" || "$bestcif" == "None" ]]; then
        echo "    [warn] no prediction for $qid ($model)"; continue
      fi
      metrics="$(score_ost "$bestcif" "$ref")"
      echo "$qid,$model,$metrics,$nsamp,$rscore" >> "$RESULTS"
      echo "    $qid [$model] -> $metrics  (sample $nsamp)"
    done
  done
done

echo; echo "==> Summary (mean over targets)"
python3 - "$RESULTS" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
cols = ["lddt","dockq_ave","ics","ips","lddt_pli","ligand_rmsd"]
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    for c in cols:
        try: agg[r["model"]][c].append(float(r[c]))
        except (ValueError, KeyError): pass
def m(v): return round(sum(v)/len(v),4) if v else "-"
print(f"{'metric':<12} {'baseline':>10} {'finetuned':>10} {'delta':>8}")
for c in cols:
    b=agg.get('baseline',{}).get(c,[]); f=agg.get('finetuned',{}).get(c,[])
    mb,mf=m(b),m(f)
    d=round(mf-mb,4) if (b and f) else "-"
    print(f"{c:<12} {str(mb):>10} {str(mf):>10} {str(d):>8}")
print("\nHigher is better for lddt/dockq/ics/ips/lddt_pli; LOWER is better for ligand_rmsd.")
print(f"Full per-target table: {sys.argv[1]}")
PY

cat <<EOF

================================================================================
DONE. Interpret:
  * interface lDDT / DockQ / lDDT-PLI UP and ligand_rmsd DOWN  => fine-tune helped
  * run a few UNRELATED targets too (forgetting check) — general metrics shouldn't drop
Reference structures expected at: $REF_DIR/<query_id>.cif  (query_id = JSON key, no dots)
================================================================================
EOF
