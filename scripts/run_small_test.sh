#!/usr/bin/env bash
# =============================================================================
# run_small_test.sh — minimal END-TO-END fine-tune smoke test.
# -----------------------------------------------------------------------------
# Goal: prove the whole pipeline works (data -> a few fine-tune steps ->
#       predict -> outputs exist) using the SMALLEST possible footprint.
#       NOT for science. Designed for a 12 GB GPU, with a CPU fallback.
#
# Stages:
#   0) sanity: env + checkpoint present
#   1) tiny data prep: 2 train + 1 val small protein-ligand complexes
#   2) fine-tune ~8 steps with finetune_test_12gb.yml
#   3) predict on the held-out complex with the fine-tuned checkpoint
#   4) verify a predicted .cif + confidence json were written -> PASS/FAIL
#
# USAGE:
#   bash run_small_test.sh                # GPU attempt
#   DEVICE=cpu bash run_small_test.sh     # guaranteed-to-run CPU functional test
# =============================================================================
set -euo pipefail

# ----------------------------- CONFIG (edit me) ------------------------------
OF3_REPO="${OF3_REPO:-$HOME/openfold-3}"
WORK="${WORK:-$HOME/of3_smoke_test}"
CKPT="${CKPT:-$HOME/.openfold3/of3-p2-155k.pt}"   # baseline weights to fine-tune from
DEVICE="${DEVICE:-gpu}"                            # gpu | cpu
NWORKERS="${NWORKERS:-1}"
# Tiny set: small protein–ligand complexes (override with your own small IDs).
TRAIN_IDS=(1stp 4hhb)        # 2 small complexes for the (toy) training set
VAL_IDS=(2hhb)               # 1 held-out complex
USE_MSA_SERVER="${USE_MSA_SERVER:-True}"          # ColabFold server (no local DBs)
# -----------------------------------------------------------------------------

S="$OF3_REPO/scripts"
CIF="$WORK/cifs"; PRE="$WORK/preprocessed"; MSA="$WORK/msas"; ALN="$WORK/alignment_arrays"
CACHE="$WORK/dataset_caches"; REPS="$WORK/MSA_representatives.fasta"
CCD="$WORK/components.cif"; BCIF="$WORK/components.bcif"
TRAIN_OUT="$WORK/train_out"; PRED_OUT="$WORK/pred_out"
mkdir -p "$CIF" "$PRE" "$MSA" "$ALN" "$CACHE" "$TRAIN_OUT" "$PRED_OUT"

echo "==> [0] sanity checks"
command -v run_openfold >/dev/null || { echo "FAIL: run_openfold not on PATH (activate the openfold3 env)"; exit 1; }
[[ -f "$CKPT" ]] || { echo "FAIL: checkpoint $CKPT not found. Run setup_openfold or fix CKPT."; exit 1; }
[[ -d "$S" ]]   || { echo "FAIL: scripts dir $S not found. Fix OF3_REPO."; exit 1; }
echo "    ok: run_openfold, checkpoint, repo scripts present"

echo "==> [1] tiny data prep ($((${#TRAIN_IDS[@]}+${#VAL_IDS[@]})) structures)"
for id in "${TRAIN_IDS[@]}" "${VAL_IDS[@]}"; do
  lc=$(echo "$id" | tr 'A-Z' 'a-z')
  [[ -f "$CIF/$lc.cif" ]] || { echo "    fetch $lc.cif"; wget -q "https://files.rcsb.org/download/$lc.cif" -O "$CIF/$lc.cif"; }
done
if [[ ! -f "$BCIF" ]]; then
  wget -q "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz" -O "$WORK/components.cif.gz"
  gunzip -f "$WORK/components.cif.gz"
  python "$S/data_preprocessing/preprocess_ccd_biotite.py" "$CCD" "$BCIF"
fi
python "$S/data_preprocessing/preprocess_pdb_of3.py" \
  --cif-dir "$CIF" --ccd-path "$CCD" --biotite-ccd-path "$BCIF" \
  --out-dir "$PRE" --num-workers "$NWORKERS" --output-format npz
META="$PRE/metadata.json"; STRUCT="$PRE/structure_files"; REFM="$PRE/reference_mols"

echo "    MSAs (ColabFold server=$USE_MSA_SERVER) — small set, should be quick"
# Collect query FASTAs and run the snakemake MSA pipeline if you have DBs; for a
# pure smoke test you can instead reuse precomputed MSAs. Minimal path:
cat "$STRUCT"/*/*.fasta > "$WORK/all_queries.fasta" 2>/dev/null || true
python "$S/utils/generate_representatives_from_msa_directory.py" \
  --msa-directory "$MSA" --out-fasta "$REPS" --protein-dbs "uniref90,uniprot,mgnify,bfd" --ncores "$NWORKERS" || \
  echo "    (representatives step needs MSA folders in $MSA — see SMALL_TEST_12GB.md if this errors)"
python "$S/data_preprocessing/preparse_alignments_of3.py" \
  --alignments_directory "$MSA" --alignment_array_directory "$ALN" \
  --max_seq_counts '{"uniref90_hits":2000,"uniprot_hits":2000,"mgnify_hits":1000,"bfd_uniref_hits":1000}' \
  --num_workers "$NWORKERS" || echo "    (preparse needs MSA folders; see plan)"

python "$S/data_preprocessing/create_pdb-weighted_training_dataset_cache.py" \
  --metadata-cache "$META" --preprocessed-dir "$STRUCT" --alignment-representatives-fasta "$REPS" \
  --output "$CACHE/training_cache.json" --dataset-name "PDB-weighted" \
  --max-resolution 9.0 --allow-missing-alignment --log-level INFO
python "$S/data_preprocessing/create_pdb_validation_dataset_cache.py" \
  --metadata-cache "$META" --preprocessed-dir "$STRUCT" --train-dataset-cache "$CACHE/training_cache.json" \
  --alignment-representatives-fasta "$REPS" --output "$CACHE/validation_cache.json" \
  --dataset-name "PDB-val" --max-resolution 9.0 --allow-missing-alignment --log-level INFO

echo "==> [2] generate a runner YAML with your paths + chosen device"
RUNNER="$WORK/runner_test.yml"
python3 - "$RUNNER" "$CKPT" "$DEVICE" "$ALN" "$CACHE" "$STRUCT" "$REFM" <<'PY'
import sys, textwrap
runner, ckpt, device, aln, cache, struct, refm = sys.argv[1:8]
y = f"""\
experiment_settings:
  mode: train
  output_dir: {runner.rsplit('/',1)[0]}/train_out
  seed: 42
  restart_checkpoint_path: {ckpt}
  ckpt_load_settings: {{manual_checkpoint_loading: true, init_from_ema_weights: false, restore_lr_scheduler: false, restore_time_step: false, strict_loading: false}}
data_module_args: {{batch_size: 1, num_workers: 1, epoch_len: 8}}
logging_config: {{log_lr: true, wandb_config: null}}
pl_trainer_args: {{accelerator: {device}, devices: 1, num_nodes: 1, precision: bf16-mixed, max_epochs: 1, log_every_n_steps: 1}}
checkpoint_config: {{every_n_epochs: 1, auto_insert_metric_name: false, save_last: true, save_top_k: -1}}
model_update:
  presets: [train]
  custom:
    settings:
      model_selection_weight_scheme: fine_tuning
      clear_cache_between_steps: true
      optimizer: {{learning_rate: 0.0003}}
      lr_scheduler: {{warmup_no_steps: 2}}
      ema: {{decay: 0.99}}
    architecture:
      shared: {{num_recycles: 1, diffusion: {{no_samples: 1}}}}
      loss_module: {{diffusion: {{chunk_size: 1}}}}
dataset_configs:
  train:
    weighted-pdb:
      dataset_class: WeightedPDBDataset
      weight: 1.0
      config:
        debug_mode: true
        template: {{n_templates: 0, take_top_k: false}}
        crop: {{token_crop: {{enabled: true, token_budget: 128, crop_weights: {{contiguous: 0.2, spatial: 0.4, spatial_interface: 0.4}}}}, chain_crop: {{enabled: true}}}}
        loss: {{loss_weights: {{bond: 4.0, smooth_lddt: 0.0}}}}
  validation:
    val-weighted-pdb:
      dataset_class: ValidationPDBDataset
      config:
        debug_mode: true
        msa: {{subsample_main: false}}
        template: {{n_templates: 0, take_top_k: false}}
        crop: {{token_crop: {{enabled: true, token_budget: 128}}}}
dataset_paths:
  weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none
    alignment_array_directory: {aln}
    dataset_cache_file: {cache}/training_cache.json
    target_structures_directory: {struct}
    target_structure_file_format: npz
    reference_molecule_directory: {refm}
    template_cache_directory: none
    template_structure_array_directory: none
    template_structures_directory: none
    template_file_format: npz
    ccd_file: null
  val-weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none
    alignment_array_directory: {aln}
    dataset_cache_file: {cache}/validation_cache.json
    target_structures_directory: {struct}
    target_structure_file_format: npz
    reference_molecule_directory: {refm}
    template_cache_directory: none
    template_structure_array_directory: none
    template_structures_directory: none
    template_file_format: npz
    ccd_file: null
"""
open(runner,"w").write(y)
print("wrote", runner)
PY

echo "==> [3] fine-tune ~8 steps (device=$DEVICE)"
run_openfold train --runner-yaml "$RUNNER" --seed 42

FT_CKPT="$TRAIN_OUT/checkpoints/last.ckpt"
[[ -f "$FT_CKPT" ]] || FT_CKPT=$(ls -t "$TRAIN_OUT"/checkpoints/*.ckpt 2>/dev/null | head -1 || true)
[[ -n "${FT_CKPT:-}" && -e "$FT_CKPT" ]] || { echo "FAIL: no checkpoint written to $TRAIN_OUT/checkpoints"; exit 1; }
echo "    fine-tuned checkpoint: $FT_CKPT"

echo "==> [4] predict on the held-out complex with the fine-tuned weights"
# Build a minimal query JSON from the val structure's sequence(s):
VID=$(echo "${VAL_IDS[0]}" | tr 'A-Z' 'a-z')
QJSON="$WORK/val_query.json"
python3 - "$STRUCT/$VID" "$QJSON" "$VID" <<'PY'
import sys, glob, json, os
sdir, out, vid = sys.argv[1:4]
seqs=[]
for fa in glob.glob(os.path.join(sdir, "*.fasta")) or glob.glob(os.path.join(sdir, "..", vid, "*.fasta")):
    cur=[]
    for line in open(fa):
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur=[]
        else: cur.append(line.strip())
    if cur: seqs.append("".join(cur))
chains=[{"molecule_type":"protein","chain_ids":[chr(65+i)],"sequence":s} for i,s in enumerate(seqs) if s]
json.dump({"queries":{f"test_{vid}":{"chains":chains}}}, open(out,"w"), indent=2)
print("wrote", out, "with", len(chains), "protein chain(s)")
PY

run_openfold predict \
  --query-json "$QJSON" \
  --inference-ckpt-path "$FT_CKPT" \
  --num-diffusion-samples 1 \
  --use-msa-server "$USE_MSA_SERVER" \
  --use-templates False \
  --output-dir "$PRED_OUT"

echo "==> [5] verify outputs"
N=$(find "$PRED_OUT" -name "*_model.cif" | wc -l)
C=$(find "$PRED_OUT" -name "*_confidences_aggregated.json" | wc -l)
echo "    predicted structures: $N    confidence jsons: $C"
if [[ "$N" -ge 1 && "$C" -ge 1 ]]; then
  echo "================= SMOKE TEST PASSED ================="
  echo "Pipeline ran end-to-end: data -> fine-tune -> predict -> outputs."
  find "$PRED_OUT" -name "*_model.cif" | head
else
  echo "================= SMOKE TEST FAILED ================="
  echo "No predicted structure produced. Check the logs above (likely OOM or data)."
  exit 1
fi
