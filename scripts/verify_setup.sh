#!/usr/bin/env bash
# =============================================================================
# verify_setup.sh — prove the OpenFold3 install is READY before you fine-tune.
# -----------------------------------------------------------------------------
# Runs a documented checklist and a real end-to-end SMOKE TEST (predicts the
# tiny ubiquitin example, 76 residues), then prints a PASS/FAIL summary.
#
# WHAT THIS VERIFIES (and why each matters):
#   1. pixi >= 0.68        - the openfold-3 workspace refuses older pixi
#                            (a fresh `curl | sh` can install 0.63; run
#                             `pixi self-update` to fix).
#   2. repo + scripts/     - the cloned openfold-3 tree the kit drives.
#   3. run_openfold        - the CLI entry point; only exists once the pixi
#                            env is installed (`pixi run -e <env> ...`).
#   4. model weights       - of3-p2-155k.pt (~2.2 GB) downloaded by
#                            `setup_openfold` to ~/.openfold3/.
#   5. CUDA toolkit         - the pixi env SHIPS its own nvcc + cutlass and
#                            sets CUDA_HOME itself. Do NOT export a system
#                            CUDA_HOME over it: pointing the DeepSpeed evoformer
#                            kernel build at a mismatched system CUDA (e.g. 12.0
#                            vs the env's 12.9) makes it fail to JIT-compile.
#                            (The "export CUDA_HOME" advice in the plan applies
#                             only to pip/conda installs, not this pixi env.)
#   6. GPU + memory        - reports card + VRAM; warns if < 24 GB (full
#                            fine-tune needs ~80 GB; < 24 GB = smoke test only).
#   7. Docker + OpenStructure (optional) - needed only for evaluate.sh scoring.
#   8. SMOKE TEST          - run_openfold predict on ubiquitin, TEMPLATES OFF,
#                            MSAs from ColabFold (the kit's policy), 1 sample.
#                            Confirms weights load + a structure file is written.
#
# USAGE:
#   bash verify_setup.sh            # full check incl. smoke test
#   SKIP_SMOKE=1 bash verify_setup.sh   # checklist only (no GPU prediction)
#
# Override defaults via env vars if your paths differ:
#   OF3_REPO=~/openfold-3  PIXI_ENV=openfold3-cuda12  CKPT=~/.openfold3/of3-p2-155k.pt
# =============================================================================
set -uo pipefail

# ----------------------------- config (overridable) --------------------------
OF3_REPO="${OF3_REPO:-$HOME/openfold-3}"
PIXI_ENV="${PIXI_ENV:-openfold3-cuda12}"
CKPT="${CKPT:-$HOME/.openfold3/of3-p2-155k.pt}"
PIXI_BIN="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"
SMOKE_OUT="${SMOKE_OUT:-$HOME/openfold3_run/smoke_test}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

# ----------------------------- pretty output ---------------------------------
PASS=0; WARN=0; FAIL=0
ok()   { echo "  [ OK ]  $*"; PASS=$((PASS+1)); }
warn() { echo "  [WARN]  $*"; WARN=$((WARN+1)); }
bad()  { echo "  [FAIL]  $*"; FAIL=$((FAIL+1)); }
hdr()  { echo; echo ">>> $*"; }

# run_openfold either directly (inside `pixi shell`) or via `pixi run -e ENV`.
if command -v run_openfold >/dev/null 2>&1; then
  RUN_OF=(run_openfold)
else
  RUN_OF=("$PIXI_BIN" run -e "$PIXI_ENV" run_openfold)
fi

echo "============================================================"
echo " OpenFold3 setup verification"
echo "   repo : $OF3_REPO"
echo "   env  : $PIXI_ENV"
echo "   ckpt : $CKPT"
echo "============================================================"

# 1. pixi version --------------------------------------------------------------
hdr "1. pixi >= 0.68"
if [[ -x "$PIXI_BIN" ]] || command -v pixi >/dev/null 2>&1; then
  PIXI_BIN="$(command -v pixi || echo "$PIXI_BIN")"
  PV="$("$PIXI_BIN" --version 2>/dev/null | awk '{print $2}')"
  # numeric compare: major*1000+minor
  if [[ -n "$PV" ]] && awk -v v="$PV" 'BEGIN{split(v,a,".");exit !((a[1]*1000+a[2])>=68)}'; then
    ok "pixi $PV"
  else
    bad "pixi $PV is too old (need >= 0.68). Fix: $PIXI_BIN self-update"
  fi
else
  bad "pixi not found. Install: curl -fsSL https://pixi.sh/install.sh | sh"
fi

# 2. repo ----------------------------------------------------------------------
hdr "2. openfold-3 repo"
if [[ -d "$OF3_REPO/scripts" && -f "$OF3_REPO/pixi.toml" ]]; then
  ok "repo at $OF3_REPO"
else
  bad "no openfold-3 at $OF3_REPO. Clone: git clone https://github.com/aqlaboratory/openfold-3.git $OF3_REPO"
fi

# 3. run_openfold --------------------------------------------------------------
hdr "3. run_openfold CLI"
if (cd "$OF3_REPO" 2>/dev/null && "${RUN_OF[@]}" --help) >/dev/null 2>&1; then
  ok "run_openfold responds (${RUN_OF[*]})"
else
  bad "run_openfold not runnable. Install env: (cd $OF3_REPO && $PIXI_BIN run -e $PIXI_ENV setup_openfold)"
fi

# 4. weights -------------------------------------------------------------------
hdr "4. model weights"
if [[ -f "$CKPT" ]]; then
  ok "weights present ($(du -h "$CKPT" | cut -f1)) at $CKPT"
else
  bad "weights missing at $CKPT. Run setup_openfold (choose default checkpoint)."
fi

# 5. CUDA toolkit (inside the env) --------------------------------------------
hdr "5. CUDA toolkit (env-provided)"
# Ask the ENV what nvcc/cutlass it has — that's what the kernel build uses.
ENV_NVCC="$( (cd "$OF3_REPO" 2>/dev/null && "$PIXI_BIN" run -e "$PIXI_ENV" bash -c 'command -v nvcc && nvcc --version | grep -oE "release [0-9.]+" | head -1') 2>/dev/null | tail -1)"
ENV_ROOT="$OF3_REPO/.pixi/envs/$PIXI_ENV"
if [[ -x "$ENV_ROOT/bin/nvcc" ]]; then
  ok "env ships its own nvcc ($ENV_NVCC)"
else
  warn "env nvcc not found at $ENV_ROOT/bin/nvcc (kernel JIT may fail)."
fi
if [[ -f "$ENV_ROOT/include/cutlass/cutlass.h" ]]; then
  ok "cutlass headers present in env (needed for evoformer_attn kernel)"
else
  warn "cutlass headers not found in env; evoformer_attn kernel may fail to build."
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  warn "CUDA_HOME=$CUDA_HOME is set in your shell. Unset it before using the pixi env -- a system CUDA can break the evoformer kernel build (must match the env's CUDA). Run:  unset CUDA_HOME"
else
  ok "CUDA_HOME not overridden (correct for the pixi env)"
fi

# 6. GPU + memory --------------------------------------------------------------
hdr "6. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  GPU_MIB="$(echo "$GPU_LINE" | grep -oE '[0-9]+ MiB' | grep -oE '[0-9]+' | head -1)"
  ok "GPU: $GPU_LINE"
  if [[ -n "$GPU_MIB" && "$GPU_MIB" -lt 24000 ]]; then
    warn "Only ${GPU_MIB} MiB VRAM. Full fine-tune wants ~80 GB; <24 GB => use GPU_PROFILE=small / run_small_test.sh only."
  fi
else
  warn "nvidia-smi not found (no NVIDIA GPU?). Prediction/training need a CUDA GPU."
fi

# 7. Docker + OpenStructure (optional, for evaluate.sh) ------------------------
hdr "7. Docker + OpenStructure (optional, eval scoring only)"
if command -v docker >/dev/null 2>&1; then
  if docker images 2>/dev/null | grep -qi openstructure; then
    ok "OpenStructure image present (eval scoring ready)"
  else
    warn "Docker present but OpenStructure image not pulled. eval scoring will be skipped. Fix: docker pull registry.scicore.unibas.ch/schwede/openstructure:latest"
  fi
else
  warn "Docker not found. evaluate.sh scoring will be skipped (training/prediction still work)."
fi

# 8. SMOKE TEST: predict the tiny ubiquitin example ----------------------------
hdr "8. Smoke test (run_openfold predict: ubiquitin, templates OFF, ColabFold MSA, 1 sample)"
if [[ "$SKIP_SMOKE" == "1" ]]; then
  warn "SKIP_SMOKE=1 set -> smoke test skipped."
elif [[ ! -f "$CKPT" ]]; then
  bad "skipping smoke test: weights missing (see step 4)."
else
  QJSON="$OF3_REPO/examples/example_inference_inputs/query_ubiquitin.json"
  [[ -f "$QJSON" ]] || bad "ubiquitin example not found at $QJSON"
  if [[ -f "$QJSON" ]]; then
    # IMPORTANT: do NOT set CUDA_HOME here. The pixi env provides its own
    # matched nvcc/cutlass; a stray system CUDA_HOME breaks the kernel build.
    unset CUDA_HOME
    rm -rf "$SMOKE_OUT" 2>/dev/null; mkdir -p "$SMOKE_OUT"
    echo "  running prediction (this can take a few minutes incl. MSA fetch)..."
    LOG="$SMOKE_OUT/_predict.log"
    if (cd "$OF3_REPO" && "${RUN_OF[@]}" predict \
          --query-json "$QJSON" \
          --inference-ckpt-path "$CKPT" \
          --use-msa-server True --use-templates False \
          --num-diffusion-samples 1 --num-model-seeds 1 \
          --output-dir "$SMOKE_OUT") >"$LOG" 2>&1; then
      # success = at least one structure file written
      STRUCT="$(find "$SMOKE_OUT" -type f \( -name '*.cif' -o -name '*.pdb' \) 2>/dev/null | head -1)"
      if [[ -n "$STRUCT" ]]; then
        ok "prediction wrote a structure: $STRUCT"
      else
        bad "predict exited 0 but no .cif/.pdb under $SMOKE_OUT (see $LOG)"
      fi
    else
      bad "predict failed. Last lines of $LOG:"
      tail -8 "$LOG" | sed 's/^/        /'
    fi
  fi
fi

# ----------------------------- summary ---------------------------------------
echo
echo "============================================================"
echo " SUMMARY:  $PASS passed, $WARN warnings, $FAIL failed"
echo "============================================================"
if [[ "$FAIL" -gt 0 ]]; then
  echo "Setup is NOT ready. Fix the [FAIL] items above (see docs: https://recep2244.github.io/openfold3-finetune-kit/troubleshooting/)."
  exit 1
elif [[ "$WARN" -gt 0 ]]; then
  echo "Setup works, with caveats above (e.g. small GPU = smoke test only, or no Docker)."
  exit 0
else
  echo "Setup is READY. You can run:  bash run_all.sh"
  exit 0
fi
