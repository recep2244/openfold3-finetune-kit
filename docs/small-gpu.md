# Small fine-tune test on a 12 GB GPU

*Companion to [Background & rationale](background.md). The goal here is narrow: prove the whole fine-tune pipeline runs end-to-end on modest hardware. All settings are verified against the `aqlaboratory/openfold-3` source.*

---

## 0. Read this first — expectations

12 GB is **below OpenFold3's documented minimum** (inference needs ≥ 32 GB; training is heavier still because you also hold Adam optimizer state + an EMA copy of the weights + activations). So be clear about what "test" means:

- ✅ **Realistic goal:** a *functional smoke test* — confirm data prep → a handful of fine-tune steps → prediction → outputs all run without crashing, and that your data is in the right format.
- ❌ **Not realistic on 12 GB:** a scientifically meaningful fine-tune (the PDE10A recipe ran on an 80 GB H100). Don't judge model quality from this test.

There are two tiers. Try Tier A; if VRAM won't cooperate, Tier B is guaranteed to run.

| Tier | Where | What it proves | Speed |
|---|---|---|---|
| **A** | 12 GB GPU, all memory knobs maxed | Real GPU training+inference path works | minutes–tens of min for ~8 steps |
| **B** | CPU (`accelerator: cpu`) | Pipeline/data/code correctness, VRAM-independent | slow (steps take minutes), but it *will* finish |

---

## 1. One-command path

Everything below is automated by **`run_small_test.sh`**:

```bash
# GPU attempt (Tier A):
OF3_REPO=$HOME/openfold-3 CKPT=$HOME/.openfold3/of3-p2-155k.pt bash run_small_test.sh

# Guaranteed functional test (Tier B):
DEVICE=cpu OF3_REPO=$HOME/openfold-3 bash run_small_test.sh
```

It: (0) checks env + checkpoint, (1) preprocesses 2 tiny train + 1 val complexes, (2) writes a runner YAML for your chosen device, (3) fine-tunes ~8 steps, (4) predicts on the held-out complex with the fine-tuned checkpoint, (5) verifies a `_model.cif` + confidence JSON exist and prints **PASS/FAIL**.

If you'd rather drive it by hand, use **`finetune_test_12gb.yml`** (the same settings as a standalone config) and the steps in Section 4.

---

## 2. Why these settings fit (the memory levers, from source)

Training VRAM ≈ params + Adam state (≈2× params) + EMA (≈1× params) + activations. We can't shrink the first three much on a single GPU except by **offloading the optimizer to CPU** (DeepSpeed ZeRO-2 offload). Activations we shrink hard:

| Lever | Default | Test value | Source / effect |
|---|---|---|---|
| `presets: [train]` | — | on | `model_setting_presets.yml`: sets `blocks_per_ckpt: 1` + `ckpt_intermediate_steps: true` → **activation checkpointing on** |
| `architecture.shared.diffusion.no_samples` | **48** | **1** | biggest single saving — diffusion samples dominate activation memory |
| `architecture.shared.num_recycles` | 3 | 1 | fewer trunk passes |
| `token_crop.token_budget` | 384–768 | **128** (→96 if needed) | crops the structure to few tokens |
| `precision` | 32-true | `bf16-mixed` | halves activation/param memory |
| `clear_cache_between_steps` | false | **true** | frees CUDA cache each step |
| `memory.train.msa_module.swiglu_seq_chunk_size` | null | 256 | chunks MSA SwiGLU activations |
| `loss_module.diffusion.chunk_size` | 4 | 1 | smaller diffusion-loss chunks |
| `batch_size` | 1 | **1** | only value supported (`runner.py`) — do not raise |
| DeepSpeed ZeRO-2 + `offload_optimizer:cpu` | — | optional | `deepspeed_configs/ds_stage2_default_offload.json` → Adam state to CPU RAM, frees several GB VRAM |

Validation is also cropped (`token_budget: 128`) — by default validation runs on **full** structures and will OOM a 12 GB card otherwise.

---

## 3. The OOM ladder (do these in order if you hit CUDA OOM)

1. **Lower the crop:** `token_budget: 128 → 96 → 64`.
2. **Confirm the savers are active:** `presets: [train]`, `no_samples: 1`, `num_recycles: 1`, `precision: bf16-mixed`, `clear_cache_between_steps: true`.
3. **Offload the optimizer to CPU** (needs decent system RAM, ~16 GB+): in the YAML set
   `pl_trainer_args.deepspeed_config_path: <repo>/deepspeed_configs/ds_stage2_default_offload.json`.
   (If DeepSpeed complains about precision, set `"bfloat16": {"enabled": true}` in that JSON and remove `precision: bf16-mixed`, or vice-versa — let one own precision.)
4. **Shrink the inputs:** use smaller complexes (short peptide + ligand) so even an uncropped pass is tiny.
5. **Fall back to CPU:** `accelerator: cpu` (Tier B). Slow but VRAM-proof — this is the definitive "is my pipeline correct?" test.
6. **Or rent a GPU:** a single 24–80 GB cloud GPU for ~1 hour runs the real low-n recipe; 12 GB is only ever a smoke test.

Reduce `nvidia-smi` background usage too (close other CUDA processes); on Windows/WSL the desktop compositor can eat ~1 GB.

---

## 4. Manual steps (if not using the script)

```bash
# 1) tiny data (2 train + 1 val) — see prepare_data.sh; set TRAIN_IDS/VAL_IDS small
TRAIN_IDS="1stp 4hhb" VAL_IDS="2hhb" WORK=$HOME/of3_smoke_test bash prepare_data.sh

# 2) point finetune_test_12gb.yml dataset_paths at $WORK/... and set restart_checkpoint_path

# 3) fine-tune ~8 steps
run_openfold train --runner-yaml finetune_test_12gb.yml --seed 42
#    -> checkpoint at ./test_finetune_output/checkpoints/last.ckpt

# 4) predict with the fine-tuned weights (loads EMA directly, no conversion)
run_openfold predict --query-json val_query.json \
  --inference-ckpt-path ./test_finetune_output/checkpoints/last.ckpt \
  --num-diffusion-samples 1 --use-templates False --output-dir ./test_pred
```

> Pick your own tiny structures if `1stp/4hhb/2hhb` aren't ideal — any small complex works; smaller = less memory. Keep query IDs **dot-free** (issue #176).

---

## 5. Pass / fail criteria

**PASS** = all of:
- preprocessing wrote `metadata.json`, per-structure `*.npz`, and `reference_mols/*.sdf`;
- training reached step > 0 and saved `checkpoints/last.ckpt` (watch the loss print — it just needs to be finite and decreasing-ish over 8 steps, not good);
- prediction wrote at least one `*_model.cif` + `*_confidences_aggregated.json`.

`run_small_test.sh` checks the last two automatically and prints **SMOKE TEST PASSED/FAILED**.

**Not a pass criterion:** structure accuracy. 8 steps on 2 toy complexes won't improve anything — you're testing plumbing, not science.

---

## 6. Common failure modes (from the issue tracker + code)

- **CUDA OOM** → walk the Section 3 ladder.
- **`DataLoader worker exited`** (#149) → keep `num_workers: 1` (already set).
- **Query ID error** (#176) → no dots in query/PDB JSON keys.
- **MSA format / chain-level MSA errors** (#188, #172) → per-chain MSA folders must contain the expected db files (`uniref90_hits`, `uniprot_hits`, `mgnify_hits`, `bfd_uniref_hits`); for a pure smoke test you can reuse precomputed MSAs or the ColabFold server.
- **`Unable to JIT load evoformer_attn` / `nvcc not found`** → set `CUTLASS_PATH` and `CUDA_HOME` (main plan §3).
- **Ligand conformer/geometry oddities** (#162, #136) → fine for a smoke test; inspect SDFs if scoring later.

---

## 7. After the smoke test passes

Move to a ≥ 24 GB GPU (cloud is cheapest) and run the real recipe: `prepare_data.sh` (your real ~10/17 split) → `finetune_lowN_single_gpu.yml` (~350 steps) → `evaluate.sh`. The exact same commands scale up — only the hardware and `token_budget`/`no_samples`/step-count change.

---

## Sources
- Memory presets & defaults (READ DIRECTLY): `openfold3/projects/of3_all_atom/config/model_setting_presets.yml`, `.../config/model_config.py`, `deepspeed_configs/ds_stage2_default_offload.json`, `openfold3/projects/of3_all_atom/runner.py` (batch_size=1 constraint) — https://github.com/aqlaboratory/openfold-3
- Inference low-memory mode & 32 GB minimum — https://openfold-3.readthedocs.io/en/latest/Installation.html and https://openfold-3.readthedocs.io/en/latest/inference.html
- Known issues (#149, #176, #188, #172, #162, #136) — https://github.com/aqlaboratory/openfold-3/issues
