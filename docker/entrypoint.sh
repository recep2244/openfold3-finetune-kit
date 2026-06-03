#!/usr/bin/env bash
# Entry point for the openfold3-finetune-kit container.
# Ensures weights are present, then dispatches a sub-command.
#
#   verify   - quick readiness check (run_openfold + weights), then a tiny predict
#   run_all  - run the full pipeline (scripts/run_all.sh)
#   predict  - run_openfold predict (pass extra args through)
#   train    - run_openfold train (pass extra args through)
#   bash     - drop into a shell
set -euo pipefail

OPENFOLD_CACHE="${OPENFOLD_CACHE:-/weights}"
CKPT="${CKPT:-$OPENFOLD_CACHE/of3-p2-155k.pt}"
export OPENFOLD_CACHE

ensure_weights() {
  if [[ -f "$CKPT" ]]; then
    echo "weights present: $CKPT"
    return
  fi
  echo "weights not found at $CKPT — downloading default checkpoint into $OPENFOLD_CACHE ..."
  # setup_openfold is interactive: feed cache dir, param dir (default), "1"
  # (default checkpoint only), and "no" (skip integration tests).
  printf '%s\n\n1\nno\n' "$OPENFOLD_CACHE" | setup_openfold
}

cmd="${1:-verify}"; shift || true

case "$cmd" in
  verify)
    ensure_weights
    command -v run_openfold >/dev/null || { echo "FAIL: run_openfold not on PATH"; exit 1; }
    echo "run_openfold OK; weights OK. Container is ready."
    echo "Run the pipeline with:  docker compose run --rm openfold3 run_all"
    ;;
  run_all)
    ensure_weights
    exec bash /kit/scripts/run_all.sh "$@"
    ;;
  predict)
    ensure_weights
    exec run_openfold predict --inference-ckpt-path "$CKPT" --use-templates False "$@"
    ;;
  train)
    ensure_weights
    exec run_openfold train "$@"
    ;;
  bash|sh)
    exec bash "$@"
    ;;
  *)
    # pass anything else straight to run_openfold
    exec run_openfold "$cmd" "$@"
    ;;
esac
