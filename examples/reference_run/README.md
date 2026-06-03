# Reference run (drop your real results here)

This folder is a **scaffold** for a real, reproducible fine-tune so the project can show a
*measured* result instead of only the schematic expected pattern. After one full run on an
80 GB GPU (`bash scripts/run_all.sh` with `GPU_PROFILE="big"`), copy these artifacts in and
fill `RUN.md`:

| File | What to put | Source |
|---|---|---|
| `results.csv` | Per-structure baseline-vs-fine-tuned scores | `<WORK>/eval/out/results.csv` |
| `train.log` | A short excerpt (loss/EMA over steps) | training stdout / W&B export |
| `env.lock` | Exact environment | `pixi list -e openfold3-cuda12 > env.lock` |
| `RUN.md` | Run metadata (template below) | fill in by hand |

`results.template.csv` shows the exact columns `evaluate.sh` writes — replace it with your real
`results.csv`. (Note: `qc_gate.py` ranks *prediction-confidence* metrics — pLDDT/ipTM/ipSAE —
which is a separate input from these reference-based eval scores.)

## `RUN.md` template

```markdown
# Reference run — <date>
- Target: PDE10A (same-target, novel-ligand held-out)
- GPU: 1× A100-80GB / H100-80GB
- Steps: ~350 · seed: 42 · GPU_PROFILE: big
- OpenFold3 commit: <git rev of ~/openfold-3>
- Base checkpoint: of3-p2-155k.pt (sha256 <…>)
- Fine-tuned checkpoint sha256: <…>
- Held-out PDB IDs: 5SH0 5SE0 5SHR 5SJL 5SH8 5SF4 5SFG 5SE5 5SHK 5SEE

## Measured deltas (held-out mean)
| metric | baseline | fine-tuned | Δ |
|---|---|---|---|
| interface lDDT | | | |
| DockQ | | | |
| lDDT-PLI | | | |
| ligand RMSD (Å) | | | |
| global lDDT (regression check) | | | |
```

Once filled, link it from the README "Expected results" section and replace the schematic
table with the measured one.
