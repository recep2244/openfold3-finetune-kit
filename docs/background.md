# Fine-tuning OpenFold3 on Specific Structures — Complete Plan

*Written as if by an OpenFold3 maintainer. Every config key, default value, script flag, and checkpoint-loading detail below was read directly from the `aqlaboratory/openfold-3` source (cloned `main`, June 2026), the official docs, the Apheris PDE10A case study, the NVIDIA BioNeMo NIM docs, and the Apheris/IPT capability articles. Exact file/line references are inline so you can verify anything. Sources at the bottom.*

---

## 0. Why fine-tune at all (the evidence)

OpenFold3 (OF3) is the AlQuraishi Lab / OpenFold consortium's Apache-2.0 reproduction of AlphaFold3 — an all-atom **co-folding** model predicting proteins, DNA, RNA, and small-molecule ligands jointly. It is strong on public benchmarks but degrades on novel chemistry, and there's a hard number for this: the **Runs N' Poses** benchmark (the same paper OF3 benchmarks against) compiled 2,600 post-cutoff complexes and found a *near-linear decline in accuracy as structural similarity to the training set drops, falling to ~20% success on the most novel ligand poses*. Co-folding models interpolate well, extrapolate badly.

Fine-tuning closes that gap. The proof point you're reproducing: **Apheris fine-tuned the public OF3 weights on 10 PDE10A protein–ligand complexes (~350 steps, ~20 h on one H100) and corrected systematic pose errors on 17 held-out structures**, with the biggest gains at the protein–ligand interface (interface lDDT, DockQ). This is a *nudge*, not a retrain — small LR, short warmup, faster EMA, few steps, so the model adapts to your target without forgetting everything else.

For data you can't pool, the field is moving to **federated fine-tuning**: the *Federated OpenFold3 Initiative* (AbbVie, Astex, BMS, J&J, Takeda + Columbia's AlQuraishi Lab) jointly fine-tunes OF3 on proprietary complexes, exchanging only privacy-preserving updates (gradients / low-rank adapters), never raw structures. Mentioned here so you know the scaling path beyond your own data.

---

## 1. The four questions a fine-tuning workflow must answer

Adapted from the Apheris capability framework — use these to frame the project:

1. **Does the base model already work on my targets?** Benchmark public OF3 on your structures first (Section 6). If it's already good, fine-tuning has little to add.
2. **Where can it be used?** Track which checkpoint wins per target/modality.
3. **If not good enough, can I improve it?** Targeted fine-tuning on a small, curated set (Sections 4–5) — the core of this plan.
4. **Can generalization be broadened?** Federated networks when internal data can't cover the chemistry (out of scope here, noted for awareness).

---

## 2. Hardware & cost (you said cloud / undecided)

OF3 **inference**: CUDA ≥ 12.1, ≥ 32 GB GPU (tested mainly on 40 GB A100). **Fine-tuning** is heavier (activations + Adam state + EMA copy of weights).

Recommended: **a single 80 GB GPU (H100 or A100-80GB)** — this matches the proven low-n recipe (~20 h for ~350 steps). Multi-GPU only matters once you scale past a few dozen structures (Section 7).

Concrete inference runtimes (NVIDIA NIM, H100 80GB, single diffusion sample) to budget your baseline + eval passes — remember evaluation runs **5 samples per structure**, so multiply by ~5:

| Seq length | ~time / sample (H100) |
|---|---|
| ~200 res | ~12 s |
| ~400 res | ~18 s |
| ~600 res | ~22 s |
| ~900 res | ~31 s |
| ~1500 res | ~57–64 s |
| ~1900 res | ~97 s |

So evaluating 30 complexes of ~400 res at 5 samples each ≈ 30×5×18 s ≈ 45 min of GPU time. Baselining is cheap; the fine-tune run is the cost driver. Install the **cuEquivariance** and **DeepSpeed4Science Evoformer** kernels — they cut memory and add ~1.2–1.6× speed.

> Managed alternatives if data governance (not compute) is the blocker: **ApherisFold** and the **NVIDIA BioNeMo OpenFold3 NIM** both keep proprietary structures inside your own infra.

---

## 3. Phase 1 — Environment

**pixi (recommended, reproducible):**
```bash
curl -fsSL https://pixi.sh/install.sh | sh            # once; restart shell
git clone git@github.com:aqlaboratory/openfold-3.git && cd openfold-3
pixi run -e openfold3-cuda12 setup_openfold           # or cuda13 / rocm7 / cpu
```
**pip:**
```bash
conda create -n openfold3 python=3.13 && conda activate openfold3
pip install "openfold3[cuequivariance]"
setup_openfold                                        # downloads params + CCD, smoke test
```

`setup_openfold` downloads weights (~2 GB) to `~/.openfold3` and sets `$OPENFOLD_CACHE/ckpt_root`.

**Environment variables that bite people** (set before training):
```bash
export CUDA_HOME=/path/to/cuda                        # else "nvcc not found"
export CUTLASS_PATH=$(python - <<'PY'                 # else DeepSpeed "Unable to JIT load evoformer_attn"
import cutlass_library, pathlib
print(pathlib.Path(cutlass_library.__file__).resolve().parent.joinpath("source"))
PY
)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"   # if "ld: cannot find -lcurand"
```
Docker image also available: `openfoldconsortium/openfold3:stable`.

Smoke test before anything else:
```bash
run_openfold predict --query-json=examples/example_inference_inputs/query_ubiquitin.json
```

### 3.1 The checkpoint question — RESOLVED (no conversion needed)

This was the one real open question, and the source answers it definitively. **You can fine-tune directly from the released `.pt` weights — no `.pt`→`.ckpt` conversion.**

The checkpoint registry (`openfold3/entry_points/parameters.py`):

| Registry name | File | Notes |
|---|---|---|
| `openfold3-p2-155k` | `of3-p2-155k.pt` | **default**, preview-2 |
| `openfold3-p2-145k` | `of3-p2-145k.pt` | preview-2 |
| `openfold3-p1` | `of3_ft3_v1.pt` | legacy (<0.4) |

Downloaded from S3 `s3://openfold3-data/openfold3-parameters/`.

How loading works (`openfold3/core/utils/checkpoint_loading_utils.py::get_state_dict_from_checkpoint`): the released `.pt` is a *raw* state dict with no `module`/`state_dict`/`ema` keys. The loader detects this (`is_pretrained_model = "module" not in ckpt and "state_dict" not in ckpt`), prepends `model.` to each key, and initializes the EMA from those weights. The EMA decay is then overridden by your config value (`experiment_runner.py::manual_load_checkpoint`). So fine-tuning *from the public weights* needs exactly this in your YAML:

```yaml
experiment_settings:
  restart_checkpoint_path: /abs/path/to/of3-p2-155k.pt   # validator checks the file exists
  ckpt_load_settings:
    manual_checkpoint_loading: true     # REQUIRED (validator enforces it)
    init_from_ema_weights: false        # the .pt has no EMA block; raw path is used
    restore_lr_scheduler: false         # start a fresh LR schedule
    restore_time_step: false            # start global_step at 0
    strict_loading: false               # tolerate head additions (e.g. PAE) / minor key diffs
```

To instead **resume your own fine-tune** later, use `restart_checkpoint_path: last` (Lightning auto-resume), or for a manual resume from a saved `.ckpt` set `init_from_ema_weights: true` and `restore_time_step: true` (those branches read the `ema` / `loops` blocks that only exist in a full training checkpoint). The validator (`validator.py`) will reject `manual_checkpoint_loading: false` combined with any manual restore flag, and rejects manual loading with no path.

---

## 4. Phase 2 — Build the dataset (80% of the work)

The trainer never reads raw `.cif`. It reads a preprocessed bundle. Required artifacts (everything the training script consumes):

```
preprocessed/standard/
├── structure_files/<id>/<id>.npz   ← REQUIRED  Biotite AtomArray per structure
│                        <id>.fasta  ←           sequences (drive MSA generation)
├── reference_mols/<LIG>.sdf         ← REQUIRED  RDKit conformer per unique ligand
└── metadata.json                    ← REQUIRED  structure/chain/interface metadata
alignment_arrays/<id>_<chain>.npz    ← REQUIRED  preparsed MSA per unique chain
dataset_caches/
├── training_cache.json              ← REQUIRED  FINAL input to the trainer
└── validation_cache.json            ← REQUIRED  for the held-out eval set
templates/                           ← OPTIONAL  (skip for first low-n run)
```

### 4.1 Structure preprocessing
Flat dir of `.cif` files (use `scripts/data_preprocessing/download_pdb_mmcif.sh` for PDB IDs). Optionally pin the CCD to your snapshot:
```bash
wget https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz && gunzip components.cif.gz
python scripts/data_preprocessing/preprocess_ccd_biotite.py components.cif components.bcif
```
Run the core preprocessor — it's a `click` CLI with these real flags (from `preprocess_pdb_of3.py`):
```bash
python scripts/data_preprocessing/preprocess_pdb_of3.py \
  --cif-dir /data/my_cifs \
  --ccd-path components.cif \
  --biotite-ccd-path components.bcif \
  --out-dir /data/preprocessed \
  --max-polymer-chains 1000 \
  --num-workers 16 \
  --chunksize 1 \
  --output-format npz          # one of: npz | cif | bcif | pkl
  # --early-stop N             # process only first N (handy for a dry run)
```
It parses (bioassembly expansion, chain renumbering), cleans per AF3 SI §2.5.4 (drops waters/H/clashing chains, adds unresolved atoms), writes RDKit reference-mol SDFs, and emits `metadata.json`.

> 🔑 **Custom-structure caveat (the silent killer).** OF3 reconstructs each polymer's *full* sequence from the mmCIF `pdbx_seq_one_letter_code_can` field and `_entity_poly_seq` records to add unresolved residues. **Your residue IDs must match the full construct, numbered from 1, with gaps in residue IDs wherever residues are unresolved** (the docs give an explicit His-tag example). Automatic for clean PDB entries; must be verified for in-house/re-refined/renumbered structures, or atoms map to the wrong residues with no error.

### 4.2 MSAs
```bash
scripts/snakemake_msa/MSA_Snakefile                                  # JackHMMER/hhblits, AF3 protocol
python scripts/utils/generate_representatives_from_msa_directory.py  # MSA dir -> query seq map
python scripts/data_preprocessing/preparse_alignments_of3.py         # raw MSA -> fast npz
```
Trivial at 10–50 chains; the heavy step only at PDB scale. ColabFold server is an alternative to hosting databases.

### 4.3 Dataset caches (the trainer's actual input)
```bash
python scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py   # -> training_cache.json
python scripts/data_preprocessing/create_pdb_validation_dataset_cache.py          # -> validation_cache.json
```
Clustering (AF3 SI §2.5.3): proteins 40% id (MMseqs2), peptides <10 res 100%, nucleic acids 100%, ligands by 100% canonical SMILES, interfaces by sorted cluster-ID tuples. Validation cache adds homology filtering vs. train (res ≤ 4.5 Å) — this is what guarantees a *clean* held-out set.

### 4.4 Templates — optional, skip first run
`preprocess_template_structures_of3.py` + `preprocess_template_alignments_new_of3.py` (appends template IDs into `training_cache.json`). PDE10A ran **templates off**; do the same to start.

> 💡 **Pick your split here, deliberately.** Choose held-out structures that are (a) genuinely novel to OF3 (post-cutoff or otherwise unseen) and (b) where the base model is already weak. Otherwise any "improvement" is memorization, not generalization.

---

## 5. Phase 3 — Fine-tune

### 5.1 Understand the shipped staged configs (`examples/training_yamls/`)
These are the OF3 team's *own* recipes — read them as ground truth:

- **`initial_training.yml`** — from-scratch training.
- **`finetune_1.yml` / `finetune_2.yml`** — **structure-loss** fine-tuning stages. Multi-dataset (weighted PDB + protein/RNA monomer distillation + disordered PDB) with sampling weights; `token_budget: 640`; diffusion `no_samples: 32`; `bond` loss `4.0`, `smooth_lddt: 0.0`; `model_selection_weight_scheme: fine_tuning`. **This is the stage that matches the Apheris pose-correction recipe.**
- **`finetune_3.yml`** — **confidence-only** stage: `train_confidence_only: true`, EMA updates *only* the aux heads (`plddt`, `pae`, `pde`, `experimentally_resolved`, `pairformer_embedding`), `token_budget: 768`, structure losses zeroed (`mse/smooth_lddt/distogram: 0.0`, `pae: 0.0001`), PAE head enabled. Run this *after* a structure-loss fine-tune if you also want recalibrated confidence.

For a single-target protein–ligand pose fix, **base your run on `finetune_1.yml` semantics** (structure loss on), strip the distillation datasets you don't have, and apply the low-n overrides below.

### 5.2 The low-n hyperparameters — with EXACT key paths

Defaults live in `openfold3/projects/of3_all_atom/config/model_config.py` (the `settings` block). Optimizer is plain **Adam** (`runner.py:845`), not AdamW. Defaults and the Apheris overrides:

| Setting | Default (source) | Low-n value | YAML path (under `model_update.custom.settings`) |
|---|---|---|---|
| learning rate | `1.8e-3` | **`3e-4`** | `optimizer.learning_rate` |
| LR warmup steps | `1000` | **`50`** | `lr_scheduler.warmup_no_steps` |
| EMA decay | `0.999` | **`0.99`** | `ema.decay` |
| total steps | (long) | **~350** | control via `data_module_args.epoch_len` × checkpoints, or stop manually |
| Adam betas/eps | `0.9 / 0.95 / 1e-8` | keep | `optimizer.beta1/beta2/eps` |
| grad clip | per-sample, `clip_val 10.0` | keep | `gradient_clipping.*` |
| LR decay | `start_decay_after_n_steps 50000`, `decay_every_n_steps 50000`, `decay_factor 0.95` | irrelevant at 350 steps | `lr_scheduler.*` |

So the override block is literally:
```yaml
model_update:
  presets: [train]
  custom:
    settings:
      model_selection_weight_scheme: fine_tuning
      optimizer:
        learning_rate: 0.0003
      lr_scheduler:
        warmup_no_steps: 50
      ema:
        decay: 0.99
```
Templates off, standard MSAs, as in PDE10A.

### 5.3 Launch
```bash
run_openfold train --runner-yaml finetune_lowN_single_gpu.yml --seed 42
```
Checkpoints land in `{output_dir}/checkpoints/`. Resume with `restart_checkpoint_path: last`. Monitor with W&B (`logging_config.wandb_config`); watch validation metrics rise while loss stays stable.

Ready-to-edit templates ship with this plan:
- **`finetune_lowN_single_gpu.yml`** — PDE10A recipe, one GPU, loads the public `.pt` directly.
- **`finetune_single_target.yml`** — faithful `finetune_1.yml` variant with distillation sets kept (anti-forgetting).
- **`finetune_multi_gpu.yml`** — same recipe scaled to N GPUs/nodes for larger sets.
- **`prepare_data.sh`** — runs all of Phase 2 for a starter set of PDB IDs.

> 🔒 **Constraint from source (`runner.py`): only `batch_size: 1` per GPU is supported.** Scale with `devices`/`num_nodes`/grad-accumulation, never `batch_size`.

---

## 6. Phase 4 — Run the fine-tuned model, then evaluate

### 6.1 Inference with your fine-tuned checkpoint — NO conversion
Another resolved gap: the trainer writes a Lightning `.ckpt`, but inference loads it directly. `experiment_runner.py::setup` calls `get_state_dict_from_checkpoint(ckpt, init_from_ema_weights=True)`, so prediction automatically uses the **EMA-averaged fine-tuned weights** — exactly what you want. Just point `--inference-ckpt-path` at your checkpoint (a file, or a DeepSpeed checkpoint *directory*):
```bash
run_openfold predict \
  --query-json eval/queries/target.json \
  --inference-ckpt-path ./finetune_pde10a_output/checkpoints/last.ckpt \
  --num-diffusion-samples 5 --use-templates False --output-dir ./eval/finetuned
```
Real predict flags (from `run_openfold.py`): `--query-json`, `--inference-ckpt-path` *or* `--inference-ckpt-name` (e.g. `openfold3-p2-155k`), `--num-diffusion-samples`, `--num-model-seeds`, `--use-msa-server True|False`, `--use-templates True|False`, `--output-dir`. (There is no `--seeds` list flag on the CLI; seed lists live in the runner YAML's `experiment_settings.seeds`, default `[42]`.)

**Ligand input JSON** (from `examples/`): each query is a key under `queries`; chains are `protein` (with `sequence`) or `ligand` (with `smiles` **or** `ccd_codes`):
```json
{"queries": {"target1": {"chains": [
  {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "MNIFEM..."},
  {"molecule_type": "ligand",  "chain_ids": "X",   "smiles": "Cc1ccccc1"}
]}}}
```
⚠️ **Query IDs must not contain dots** (`.`) — open bug #176; a key like `pdb_7L39` is fine, `7.39` is not.

### 6.2 Output layout & picking the best sample
`writer.py` writes, per query/seed/sample:
```
<out>/<query_id>/seed_<seed>/<query_id>_seed_<seed>_sample_<s>_model.cif
<out>/<query_id>/seed_<seed>/<query_id>_seed_<seed>_sample_<s>_confidences_aggregated.json
```
The aggregated JSON contains `avg_plddt`, `gpde`, and **`sample_ranking_score`** (AF3 SI §5.9.3 ranking metric). Select the sample with the max `sample_ranking_score` — that's the "highest-confidence sample" the PDE10A protocol uses. (pLDDT is in the predicted CIF's B-factor column.)

### 6.3 Metrics & harness
Target metrics (PDE10A set): global GDT · intra-protein lDDT · intra-ligand lDDT · **protein–ligand interface lDDT** · **DockQ**.

OF3 ships native implementations in `openfold3/core/metrics/quality.py` (`lddt`, `interface_lddt`, `dockq`, `dockq_full_complex`, `gdt_ts`, `gdt_ha`, `rmsd`) — but they operate on the model's feature tensors, so for file-based scoring of arbitrary predictions use **OpenStructure (OST)**:
```bash
docker pull registry.scicore.unibas.ch/schwede/openstructure:latest
# protein / interface:
docker run --rm -v "$PWD:/home" registry.scicore.unibas.ch/schwede/openstructure:latest \
  compare-structures -m pred.cif -r reference.cif \
  --fault-tolerant --min-pep-length 4 --min-nuc-length 4 --lddt --dockq --ics --ips -o cs.json
# protein-ligand pose:
docker run --rm -v "$PWD:/home" registry.scicore.unibas.ch/schwede/openstructure:latest \
  compare-ligand-structures -m pred.cif -r reference.cif --lddt-pli --rmsd -o lig.json
```
lDDT bands: >0.90 excellent, 0.70–0.90 good, 0.50–0.70 moderate, <0.50 poor. **`evaluate.sh` automates all of 6.1–6.3** (baseline vs fine-tuned, best-sample selection, OST scoring, summary table). Plot baseline vs. fine-tuned across all held-out structures with error bars.

**Success looks like** (per PDE10A): large jumps in interface lDDT / DockQ / lDDT-PLI (and lower ligand RMSD), modest gains in global metrics, **no regression** in intra-protein accuracy.

> ⚠️ **Catastrophic-forgetting check.** Also run the fine-tuned model on a few *unrelated* targets. If general performance dropped, lower the LR, cut steps, or mix general PDB complexes into training (why `finetune_1.yml` / `finetune_single_target.yml` keep distillation datasets alongside your target).

---

## 7. Scaling beyond low-n

Multi-GPU is config-only:
```yaml
pl_trainer_args:
  devices: 8        # GPUs per node   (finetune_1/2/3 ship with devices:8, num_nodes:32)
  num_nodes: 4
  precision: bf16-mixed
```
With more data: raise step count, lengthen `warmup_no_steps` back toward 1000, raise `ema.decay` back toward 0.999, and reintroduce the distillation datasets (with `dataset_paths`) to protect generality. For privacy-constrained multi-party training, the federated route (low-rank adapters / gradient exchange) is the documented pattern.

---

## 8. Gotchas checklist (incl. items pulled from open GitHub issues)

Protocol / config:
- [x] **Train-from `.pt`** — fine-tune straight from `of3-p2-155k.pt`; `manual_checkpoint_loading: true`, `strict_loading: false`. No conversion. (3.1)
- [x] **Predict-from `.ckpt`** — point `--inference-ckpt-path` at your fine-tuned `.ckpt`; it loads EMA weights. No conversion. (6.1)
- [ ] **`batch_size: 1` only** per GPU (source-enforced); scale via devices/nodes. (5)
- [ ] **Right stage**: structure loss (finetune_1/2) for pose; confidence-only (finetune_3) only to recalibrate. (5.1)
- [ ] **Residue numbering** in custom CIFs matches the full construct with gapped IDs. (4.1)
- [ ] **CUTLASS_PATH / CUDA_HOME** set, or DeepSpeed/JIT kernels fail. (3)

Evaluation:
- [ ] **Held-out set genuinely novel** to OF3, base model weak on it. (4.4)
- [ ] **Baseline measured** with the same 5-sample / top-`sample_ranking_score` protocol. (6)
- [ ] **Forgetting check** on unrelated targets post-fine-tune. (6)
- [ ] **Data governance**: keep proprietary structures on your own infra (or ApherisFold / BioNeMo NIM). (2)

Known issues to design around (aqlaboratory/openfold-3 tracker, June 2026):
- [ ] **Query IDs with dots `.` are rejected** (#176) — keep JSON query keys dot-free.
- [ ] **MSA format / chain-level MSA features** can error (#188, #172) — match the expected per-chain MSA db filenames (`uniref90_hits`, `uniprot_hits`, `mgnify_hits`, `bfd_uniref_hits`, `hmm_output`).
- [ ] **DataLoader worker crashes** on some installs (#149) — lower `num_workers` (try `1`) if you hit it.
- [ ] **Ligand conformer init / geometry** quirks (#162 random conformer init, #136 sp2 C=C) — sanity-check generated reference-mol SDFs.
- [ ] **Template preprocessing timeout** is fixed/short (#164) — relevant only if you enable templates.
- [ ] **ROCm/AMD**: DeepSpeed Evoformer attention may break (#177) — prefer NVIDIA, or the documented ROCm Triton path.

---

## 9. First milestone (1–2 days)

1. Environment + ubiquitin smoke test. (½ day)
2. Pick ~10 train + ~15 eval structures: novel to OF3, base model weak. (½ day)
3. Run the data pipeline on those ~25, templates off. (½ day; MSAs quick at this scale)
4. `bash prepare_data.sh`, then baseline the eval set (`evaluate.sh`, baseline only). (hours)
5. Fine-tune with `finetune_lowN_single_gpu.yml`, ~350 steps. (~20 h unattended)
6. `bash evaluate.sh` (baseline vs fine-tuned, OST interface/DockQ/lDDT-PLI); plot; run forgetting check. (hours)

Interface lDDT / DockQ up without protein regression ⇒ workflow validated; scale data and targets from there.

**The full toolchain now:** `prepare_data.sh` → (`finetune_lowN_single_gpu.yml` | `finetune_single_target.yml` | `finetune_multi_gpu.yml`) → `evaluate.sh`.

---

## Sources

- OpenFold3 Training how-to — https://openfold-3.readthedocs.io/en/latest/training.html
- OpenFold3 Training Data Pipeline — https://openfold-3.readthedocs.io/en/latest/data_pipeline_reference.html
- OpenFold3 Setup / Installation — https://openfold-3.readthedocs.io/en/latest/Installation.html
- OpenFold3 Configuration Reference — https://openfold-3.readthedocs.io/en/latest/configuration_reference.html
- aqlaboratory/openfold-3 source (READ DIRECTLY): `examples/training_yamls/{initial_training,finetune_1,finetune_2,finetune_3}.yml`; `openfold3/projects/of3_all_atom/config/model_config.py` (optimizer/lr_scheduler/ema defaults); `openfold3/projects/of3_all_atom/runner.py` (Adam); `openfold3/core/utils/checkpoint_loading_utils.py` + `openfold3/entry_points/experiment_runner.py` + `validator.py` (checkpoint loading); `openfold3/entry_points/parameters.py` (checkpoint registry); `scripts/data_preprocessing/preprocess_pdb_of3.py` (CLI flags) — https://github.com/aqlaboratory/openfold-3
- Apheris — PDE10A low-n fine-tuning case study — https://www.apheris.com/resources/blog/fine-tuning-openfold3-on-a-small-set-of-structures-the-pde10-case-study
- Apheris/IPT — Building fine-tuning capabilities for co-folding models (4-question framework, Federated OF3 Initiative) — https://www.iptonline.com/articles/building-fine-tuning-capabilities-for-co-folding-models-in-pharm
- aqlaboratory/openfold-3 inference/eval source (READ DIRECTLY): `openfold3/run_openfold.py` (predict CLI); `openfold3/core/runners/writer.py` (output naming + `sample_ranking_score`); `openfold3/core/metrics/quality.py` (lddt/interface_lddt/dockq/gdt); `examples/example_inference_inputs/query_*ligand*.json` (ligand input) — https://github.com/aqlaboratory/openfold-3
- aqlaboratory/openfold-3 open issues (gotchas): #176 dots in query IDs, #188/#172 MSA format, #149 DataLoader workers, #162/#136 ligand conformer/geometry, #177 ROCm Evoformer — https://github.com/aqlaboratory/openfold-3/issues
- NVIDIA BioNeMo OpenFold3 NIM — performance + OST/lDDT eval harness — https://docs.nvidia.com/nim/bionemo/openfold3/latest/performance.html
- Runs N' Poses benchmark (novelty vs. accuracy) — https://www.biorxiv.org/content/10.1101/2025.02.03.636309v3
- OpenFold Portal — https://portal.openfold.omsf.io/fine-tuning
- OMSF — Behind the scenes: OpenFold3 design — https://omsf.substack.com/p/behind-the-scenes-openfold3-design
