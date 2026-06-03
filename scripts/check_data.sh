#!/usr/bin/env bash
# Curated by Recep Adiyaman — https://github.com/recep2244/openfold3-finetune-kit
# =============================================================================
# check_data.sh — PREFLIGHT: confirm your data is correct BEFORE training.
# -----------------------------------------------------------------------------
# Training failures are usually silent data problems. This script checks every
# required artifact exists and is non-empty, and guards the known OpenFold3
# gotchas so you find problems in seconds instead of after a long run.
#
# Checks (each prints PASS / FAIL with a plain-English reason):
#   1. preprocessed structures   <WORK>/preprocessed/structure_files/<id>/<id>.npz
#   2. reference molecules        <WORK>/preprocessed/reference_mols/*.sdf
#   3. metadata.json              valid JSON, non-empty
#   4. MSA arrays (REQUIRED)      <WORK>/alignment_arrays/*.npz   <-- most common gap
#   5. dataset caches             training_cache.json / validation_cache.json valid + non-empty
#   6. query JSONs (if present)   no dots in names (#176); no chain-level MSA keys (#172);
#                                 MSA files referenced are .a3m/.sto (#188)
#
# USAGE:  WORK=$HOME/openfold3_run bash check_data.sh
#         (run_all.sh calls this automatically after data prep)
# =============================================================================
set -uo pipefail   # NOTE: no -e; we want to run ALL checks and report them all

WORK="${WORK:-$HOME/openfold3_run}"
QUERY_DIR="${QUERY_DIR:-$WORK/eval/queries}"   # optional

PRE="$WORK/preprocessed"
STRUCT="$PRE/structure_files"
REFM="$PRE/reference_mols"; [[ -d "$REFM" ]] || REFM="$PRE/reference_molecules"
META="$PRE/metadata.json"
ALN="$WORK/alignment_arrays"
TC="$WORK/dataset_caches/training_cache.json"
VC="$WORK/dataset_caches/validation_cache.json"

PASS=0; FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo "         -> $*"; }

echo "================ OpenFold3 data preflight ================"
echo "WORK = $WORK"
echo

echo "1) Preprocessed structures (.npz)"
n_npz=$(find "$STRUCT" -name "*.npz" -size +0c 2>/dev/null | wc -l)
if [[ "$n_npz" -ge 1 ]]; then ok "$n_npz structure .npz file(s) found"
else bad "no non-empty .npz under $STRUCT"; info "did preprocess_pdb_of3.py run? (prepare_data.sh step 3)"; fi

echo "2) Reference molecules (.sdf)"
n_sdf=$(find "$REFM" -name "*.sdf" -size +0c 2>/dev/null | wc -l)
if [[ "$n_sdf" -ge 1 ]]; then ok "$n_sdf reference-molecule .sdf file(s) found in $(basename "$REFM")/"
else bad "no .sdf in $REFM"; info "ligand reference mols missing — needed for protein-ligand training"; fi

echo "3) metadata.json"
if [[ -s "$META" ]] && python3 -c "import json,sys;d=json.load(open('$META'));sys.exit(0 if d else 1)" 2>/dev/null; then
  ok "metadata.json is valid, non-empty JSON"
else bad "metadata.json missing/empty/invalid at $META"; fi

echo "4) MSA arrays (.npz) — the most commonly missing piece"
n_aln=$(find "$ALN" -name "*.npz" -size +0c 2>/dev/null | wc -l)
if [[ "$n_aln" -ge 1 ]]; then ok "$n_aln preparsed MSA .npz file(s) found"
else
  bad "no MSA .npz under $ALN"
  info "MSA step did not complete. With MSA_MODE=colabfold check internet + that the"
  info "ColabFold bridge produced per-chain folders, then re-run prepare_data.sh."
fi

echo "5) Dataset caches"
for pair in "training:$TC" "validation:$VC"; do
  name="${pair%%:*}"; path="${pair#*:}"
  if [[ -s "$path" ]] && python3 -c "import json,sys;d=json.load(open('$path'));sys.exit(0 if d else 1)" 2>/dev/null; then
    cnt=$(python3 -c "import json;d=json.load(open('$path'));print(len(d) if hasattr(d,'__len__') else 0)" 2>/dev/null || echo "?")
    ok "$name cache valid (top-level entries: $cnt)"
  else
    bad "$name cache missing/empty/invalid at $path"
  fi
done

echo "6) Query JSON checks (issues #176 / #172 / #188)"
if compgen -G "$QUERY_DIR/*.json" >/dev/null 2>&1; then
  python3 - "$QUERY_DIR" <<'PY'
import json, glob, os, sys
bad = 0
FORBIDDEN = {"use_msas", "use_main_msas", "use_paired_msas"}   # #172: not allowed at chain level
for jf in glob.glob(os.path.join(sys.argv[1], "*.json")):
    try: d = json.load(open(jf))
    except Exception as e:
        print(f"  [FAIL] {os.path.basename(jf)} is not valid JSON: {e}"); bad+=1; continue
    for qid, q in d.get("queries", {}).items():
        if "." in qid:
            print(f"  [FAIL] {os.path.basename(jf)}: query id '{qid}' contains a dot — rename (#176)"); bad+=1
        for ch in q.get("chains", []):
            hit = FORBIDDEN & set(ch.keys())
            if hit:
                print(f"  [FAIL] {os.path.basename(jf)}: chain has forbidden key(s) {sorted(hit)} — "
                      f"remove from chain level (#172)"); bad+=1
            for p in (ch.get("main_msa_file_paths") or []):
                if isinstance(p, str) and not p.endswith((".a3m", ".sto", "")) and os.path.splitext(p)[1]:
                    print(f"  [FAIL] {os.path.basename(jf)}: MSA file '{p}' must be .a3m or .sto (#188)"); bad+=1
    if bad == 0:
        pass
print("  [PASS] query JSONs look valid" if bad == 0 else f"  ({bad} query-JSON problem(s) above)")
sys.exit(1 if bad else 0)
PY
  if [[ $? -eq 0 ]]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
else
  echo "  [skip] no query JSONs in $QUERY_DIR yet (created at evaluation time)"
fi

echo
echo "================ RESULT ================"
echo "PASS: $PASS    FAIL: $FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "All good — safe to start fine-tuning."
  exit 0
else
  echo "Fix the [FAIL] items above before training (docs: https://recep2244.github.io/openfold3-finetune-kit/troubleshooting/)."
  exit 1
fi
