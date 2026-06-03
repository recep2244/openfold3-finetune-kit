#!/usr/bin/env bash
# =============================================================================
# run_all.sh — end-to-end OpenFold3 target fine-tuning pipeline.
# -----------------------------------------------------------------------------
# Edit the CONFIG block below, then run:  bash scripts/run_all.sh
#
# Runs in order, stopping with an actionable message if any stage fails:
#   [A] readiness check (tools + model weights)
#   [B] data preparation (download structures + ColabFold MSAs)
#   [C] fine-tune on the TRAIN structures
#   [D] evaluate baseline vs fine-tuned on the HELD-OUT structures
#
# Docs: https://recep2244.github.io/openfold3-finetune-kit/
# =============================================================================
set -euo pipefail

# ============================ CONFIG — EDIT THESE =============================
# 1) Folder where you cloned OpenFold3 (it contains a 'scripts' folder):
OF3_REPO="$HOME/openfold-3"

# 2) A folder where ALL your results will be saved (created automatically):
WORK="$HOME/openfold3_run"

# 3) The downloaded model weights file (created by 'setup_openfold'):
CKPT="$HOME/.openfold3/of3-p2-155k.pt"

# 4) Your TRAIN structures (4-character PDB codes), separated by spaces:
TRAIN_IDS="5SDY 5SIQ 5SI7 5SIG 5SI5 5SI8 5SIY 5SG5 5SGL 5SIH"

# 5) Your HELD-OUT (test) structures — the model never trains on these:
VAL_IDS="5SH0 5SE0 5SHR 5SJL 5SH8 5SF4 5SFG 5SE5 5SHK 5SEE"

# 6) Which GPU size are you on?  "big" (>=24GB) or "small" (12GB smoke test)
GPU_PROFILE="big"
# =============================================================================

HERE="$(cd "$(dirname "$0")" && pwd)"     # the scripts/ folder (sibling .sh live here)
REPO_ROOT="$(cd "$HERE/.." && pwd)"       # repo root; configs live in $REPO_ROOT/configs
CONFIGS="$REPO_ROOT/configs"
say() { echo; echo "============================================================"; echo ">>> $*"; echo "============================================================"; }
die() { echo; echo "!!! STOPPED: $*"; echo "    Troubleshooting: https://recep2244.github.io/openfold3-finetune-kit/troubleshooting/"; exit 1; }

# ---------- [A] readiness check ----------
say "[A] Checking your computer is ready"
command -v run_openfold >/dev/null || die "'run_openfold' not found. Activate the OpenFold3 environment first (see README step 2)."
[[ -d "$OF3_REPO/scripts" ]] || die "OF3_REPO is wrong: no 'scripts' folder in $OF3_REPO. Fix line 'OF3_REPO=' above."
[[ -f "$CKPT" ]] || die "Model weights not found at $CKPT. Run 'setup_openfold' first, or fix the 'CKPT=' line."
command -v wget >/dev/null || die "'wget' is not installed. Install it (Ubuntu: sudo apt install wget)."
echo "    OK: tools found, repo found, model weights found."

# choose the fine-tune config by GPU profile
if [[ "$GPU_PROFILE" == "small" ]]; then
  FT_YAML="$CONFIGS/finetune_test_12gb.yml"
else
  FT_YAML="$CONFIGS/finetune_lowN_single_gpu.yml"
fi
[[ -f "$FT_YAML" ]] || die "Config $FT_YAML not found. Keep the configs/ folder next to scripts/."

# ---------- [B] data prep ----------
# ColabFold is used for MSAs ONLY (no templates / no structures). Templates are
# off in every config and every predict call below.
say "[B] Preparing data (structures + ColabFold MSAs only, no databases needed)"
OF3_REPO="$OF3_REPO" WORK="$WORK" MSA_MODE="colabfold" \
  TRAIN_IDS=($TRAIN_IDS) VAL_IDS=($VAL_IDS) \
  bash "$HERE/prepare_data.sh" || die "Data preparation failed (see messages above)."

# Resolve the paths prepare_data.sh produced:
ALN="$WORK/alignment_arrays"
TRAIN_CACHE="$WORK/dataset_caches/training_cache.json"
VAL_CACHE="$WORK/dataset_caches/validation_cache.json"
STRUCT="$WORK/preprocessed/structure_files"
REFM="$WORK/preprocessed/reference_mols"
[[ -d "$REFM" ]] || REFM="$WORK/preprocessed/reference_molecules"
[[ -f "$TRAIN_CACHE" ]] || die "Training cache not created at $TRAIN_CACHE."

# ---------- preflight: verify data before spending GPU time ----------
say "[B2] Verifying data is complete and correct (preflight)"
WORK="$WORK" bash "$HERE/check_data.sh" || die "Data preflight failed — fix the [FAIL] items above before training."

# ---------- write a runner YAML from the template with YOUR paths ----------
say "[C] Fine-tuning the model on your TRAIN structures"
RUNNER="$WORK/runner_finetune.yml"
python3 - "$FT_YAML" "$RUNNER" "$CKPT" "$ALN" "$TRAIN_CACHE" "$VAL_CACHE" "$STRUCT" "$REFM" "$WORK" <<'PY'
import sys, re
tmpl, out, ckpt, aln, tc, vc, struct, refm, work = sys.argv[1:10]
y = open(tmpl).read()
# fill the checkpoint + output dir
y = re.sub(r'restart_checkpoint_path:.*', f'restart_checkpoint_path: {ckpt}', y, count=1)
y = re.sub(r'output_dir:.*', f'output_dir: {work}/train_out', y, count=1)
# fill all dataset paths (both train + val blocks)
y = y.replace("/shared/openfold3/finetune/alignment_arrays", aln)
y = y.replace("/shared/openfold3/test/alignment_arrays", aln)
y = y.replace("/shared/openfold3/finetune/dataset_caches/training_cache.json", tc)
y = y.replace("/shared/openfold3/test/dataset_caches/training_cache.json", tc)
y = y.replace("/shared/openfold3/finetune/dataset_caches/validation_cache.json", vc)
y = y.replace("/shared/openfold3/test/dataset_caches/validation_cache.json", vc)
y = y.replace("/shared/openfold3/finetune/preprocessed/structure_files", struct)
y = y.replace("/shared/openfold3/test/preprocessed/structure_files", struct)
y = y.replace("/shared/openfold3/finetune/preprocessed/reference_mols", refm)
y = y.replace("/shared/openfold3/test/preprocessed/reference_mols", refm)
open(out,"w").write(y)
print("wrote", out)
PY
run_openfold train --runner-yaml "$RUNNER" --seed 42 || die "Fine-tuning failed. If it says 'out of memory', set GPU_PROFILE=\"small\" or see SMALL_TEST_12GB.md."

FT_CKPT="$WORK/train_out/checkpoints/last.ckpt"
[[ -f "$FT_CKPT" ]] || FT_CKPT=$(ls -t "$WORK"/train_out/checkpoints/*.ckpt 2>/dev/null | head -1 || true)
[[ -n "${FT_CKPT:-}" && -e "$FT_CKPT" ]] || die "Training finished but no checkpoint was saved."
echo "    Fine-tuned model: $FT_CKPT"

# ---------- [D] evaluate ----------
say "[D] Evaluating baseline vs fine-tuned on your HELD-OUT structures"
# Build per-target query JSONs + grab reference CIFs for the held-out set.
EVALQ="$WORK/eval/queries"; EVALR="$WORK/eval/references"; mkdir -p "$EVALQ" "$EVALR"
for id in $VAL_IDS; do
  lc=$(echo "$id" | tr 'A-Z' 'a-z')
  [[ -f "$EVALR/$lc.cif" ]] || cp "$WORK/cifs/$lc.cif" "$EVALR/$lc.cif" 2>/dev/null || \
     wget -q "https://files.rcsb.org/download/$lc.cif" -O "$EVALR/$lc.cif"
  # query JSON from the preprocessed sequences (protein-only is fine for scoring)
  python3 - "$STRUCT/$lc" "$EVALQ/$lc.json" "$lc" <<'PY'
import sys, glob, os, json
sdir, out, qid = sys.argv[1:4]
seqs=[]
for fa in glob.glob(os.path.join(sdir, "*.fasta")):
    cur=[]
    for line in open(fa):
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur=[]
        else: cur.append(line.strip())
    if cur: seqs.append("".join(cur))
chains=[{"molecule_type":"protein","chain_ids":[chr(65+i)],"sequence":s} for i,s in enumerate(dict.fromkeys(seqs)) if s]
json.dump({"queries":{qid:{"chains":chains}}}, open(out,"w"), indent=2)
PY
done

QUERY_DIR="$EVALQ" REF_DIR="$EVALR" OUT="$WORK/eval/out" \
  FT_CKPT="$FT_CKPT" BASELINE_CKPT_NAME="openfold3-p2-155k" NSAMPLES=5 \
  bash "$HERE/evaluate.sh" || die "Evaluation failed (often a Docker/OpenStructure issue — see README)."

say "ALL DONE"
echo "Results folder:        $WORK"
echo "Fine-tuned model:      $FT_CKPT"
echo "Eval scores (CSV):     $WORK/eval/out/results.csv"
echo
echo "Read the printed summary above: if interface lDDT / DockQ / lDDT-PLI went UP"
echo "and ligand RMSD went DOWN for 'finetuned' vs 'baseline', the fine-tune helped."
