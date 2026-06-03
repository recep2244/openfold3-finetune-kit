#!/usr/bin/env bats
# Regression tests for the kit's shell scripts.
# These cover behaviour that does NOT require a GPU or OpenFold3 — syntax,
# permissions, and graceful failure when prerequisites are missing — so they
# run on any CI runner.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/.." && pwd)"
}

@test "all scripts pass 'bash -n' syntax check" {
  for s in "$REPO"/scripts/*.sh "$REPO"/docker/entrypoint.sh; do
    run bash -n "$s"
    [ "$status" -eq 0 ] || { echo "syntax error in $s"; return 1; }
  done
}

@test "all scripts are executable" {
  for s in "$REPO"/scripts/*.sh; do
    [ -x "$s" ] || { echo "not executable: $s"; return 1; }
  done
}

@test "aggregating scripts intentionally omit 'set -e'" {
  # verify_setup.sh and check_data.sh tally all checks; -e would abort early.
  grep -q 'set -uo pipefail' "$REPO/scripts/verify_setup.sh"
  grep -q 'set -uo pipefail' "$REPO/scripts/check_data.sh"
}

@test "run_all.sh aborts with guidance when run_openfold is missing" {
  # CI has no run_openfold on PATH, so the readiness check should fail clearly.
  run bash "$REPO/scripts/run_all.sh"
  [ "$status" -ne 0 ]
  [[ "$output" == *"run_openfold"* ]]
}

@test "verify_setup.sh runs and prints a PASS/FAIL summary" {
  run env SKIP_SMOKE=1 PIXI_BIN=/nonexistent bash "$REPO/scripts/verify_setup.sh"
  [[ "$output" == *"SUMMARY"* ]]
}

@test "check_data.sh reports a clear failure on a missing work dir" {
  run env WORK="$BATS_TEST_TMPDIR/empty_work" bash "$REPO/scripts/check_data.sh"
  [ "$status" -ne 0 ]
}

@test "qc_gate.py composite ranker passes its self-test" {
  run python3 "$REPO/scripts/qc_gate.py" --self-test
  [ "$status" -eq 0 ]
  [[ "$output" == *"self-test OK"* ]]
}

@test "ipsae_score.sh and foldseek_search.sh show usage without args" {
  run bash "$REPO/scripts/ipsae_score.sh"
  [ "$status" -ne 0 ]
  run bash "$REPO/scripts/foldseek_search.sh"
  [ "$status" -ne 0 ]
}
