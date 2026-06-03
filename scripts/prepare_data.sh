#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh — end-to-end OpenFold3 fine-tuning data pipeline for a
#                   STARTER SET of PDB IDs.
# -----------------------------------------------------------------------------
# Turns a handful of PDB IDs into the preprocessed bundle the OF3 trainer needs:
#   structure NPZs + reference-mol SDFs + metadata.json + per-chain MSA NPZs
#   + training_cache.json + validation_cache.json   (templates OFF)
#
# Every command/flag below was taken from the aqlaboratory/openfold-3 source:
#   scripts/data_preprocessing/preprocess_pdb_of3.py        (click CLI)
#   scripts/data_preprocessing/preprocess_ccd_biotite.py    (argparse)
#   scripts/data_preprocessing/preparse_alignments_of3.py   (click CLI)
#   scripts/utils/generate_representatives_from_msa_directory.py (argparse)
#   scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py
#   scripts/data_preprocessing/create_pdb_validation_dataset_cache.py
#   scripts/snakemake_msa/MSA_Snakefile + example_msa_config_protein.json
#
# PREREQS: openfold3 env active (pip/pixi), repo cloned, AWS CLI + wget present.
# USAGE:   bash prepare_data.sh          # uses the defaults in the CONFIG block
# =============================================================================
set -euo pipefail

# ----------------------------- CONFIG (edit me) ------------------------------
# Path to your cloned openfold-3 repo (for the scripts/ directory):
OF3_REPO="${OF3_REPO:-$HOME/openfold-3}"

# Where all preprocessed data will be written:
WORK="${WORK:-/shared/openfold3/finetune}"

# How many CPU workers for preprocessing / MSA preparsing:
NWORKERS="${NWORKERS:-16}"

# TRAIN set (gets gradient updates). Default: the Apheris PDE10A 10-complex set.
TRAIN_IDS=(5SDY 5SIQ 5SI7 5SIG 5SI5 5SI8 5SIY 5SG5 5SGL 5SIH)

# VALIDATION / held-out set (NO gradient updates). Default: PDE10A 17-complex set.
VAL_IDS=(5SH0 5SE0 5SHR 5SJL 5SH8 5SF4 5SFG 5SE5 5SHK 5SEE 5SFL 5SJU 5SKE 5SKU 5SKO 5SEA 5SKR)

# MSA generation. Choose ONE:
#   "colabfold" -> NO databases needed; uses the ColabFold web server (recommended
#                  for small sets). Filenames are auto-bridged to the
#                  names the trainer expects. Needs internet; not for sensitive data.
#   "snakemake" -> run the OF3 MSA pipeline locally (requires ~330GB+ databases)
#   "skip"      -> you already have per-chain MSA folders in $MSA_RAW
MSA_MODE="${MSA_MODE:-colabfold}"
DB_PATH="${DB_PATH:-/shared/openfold3/alignment_dbs}"   # base path of sequence DBs (snakemake mode only)

# Release-date / resolution filters for the dataset caches:
TRAIN_MAX_RES="9.0"     # AF3-style light filter for training
VAL_MAX_RES="4.5"       # stricter filter for validation
# Cutoff so the held-out set is treated as novel relative to OF3 training:
VAL_MIN_RELEASE_DATE="2022-01-01"
# -----------------------------------------------------------------------------

SCRIPTS="$OF3_REPO/scripts"
CIF_DIR="$WORK/cifs"
PRE_DIR="$WORK/preprocessed"           # preprocess_pdb_of3 output root
MSA_RAW="$WORK/msas_completed"         # per-chain MSA folders (raw)
ALN_ARR="$WORK/alignment_arrays"       # preparsed MSA npz (REQUIRED by trainer)
CACHE_DIR="$WORK/dataset_caches"
REPS_FASTA="$WORK/MSA_representatives.fasta"
CCD_CIF="$WORK/components.cif"
CCD_BCIF="$WORK/components.bcif"

mkdir -p "$CIF_DIR" "$PRE_DIR" "$MSA_RAW" "$ALN_ARR" "$CACHE_DIR"

echo "==> [1/6] Download mmCIF files for the starter set"
for id in "${TRAIN_IDS[@]}" "${VAL_IDS[@]}"; do
  lc=$(echo "$id" | tr '[:upper:]' '[:lower:]')
  if [[ ! -f "$CIF_DIR/$lc.cif" ]]; then
    echo "    fetching $lc.cif"
    wget -q "https://files.rcsb.org/download/${lc}.cif" -O "$CIF_DIR/$lc.cif"
  fi
done

echo "==> [2/6] Prepare a Biotite CCD pinned to the current PDB snapshot"
if [[ ! -f "$CCD_BCIF" ]]; then
  wget -q "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz" -O "$WORK/components.cif.gz"
  gunzip -f "$WORK/components.cif.gz"
  python "$SCRIPTS/data_preprocessing/preprocess_ccd_biotite.py" "$CCD_CIF" "$CCD_BCIF"
fi

echo "==> [3/6] Structure preprocessing (mmCIF -> NPZ + reference mols + metadata.json)"
python "$SCRIPTS/data_preprocessing/preprocess_pdb_of3.py" \
  --cif-dir "$CIF_DIR" \
  --ccd-path "$CCD_CIF" \
  --biotite-ccd-path "$CCD_BCIF" \
  --out-dir "$PRE_DIR" \
  --num-workers "$NWORKERS" \
  --output-format npz
# NOTE: inspect $PRE_DIR to confirm the exact subfolder layout produced
#       (typically: structure_files/, reference_mols|reference_molecules/, metadata.json).
META_CACHE="$PRE_DIR/metadata.json"
STRUCT_DIR="$PRE_DIR/structure_files"
REF_MOLS="$PRE_DIR/reference_mols"   # adjust if the script wrote 'reference_molecules'

echo "==> [4/6] MSAs (mode: $MSA_MODE)"
if [[ "$MSA_MODE" == "colabfold" ]]; then
  # ---- NO-DATABASE PATH: fetch MSAs from the ColabFold server ----
  # 1) Build one query JSON listing every chain of every structure (protein seqs).
  CF_QUERY="$WORK/colabfold_query.json"
  python3 - "$STRUCT_DIR" "$CF_QUERY" <<'PY'
import sys, glob, os, json
sdir, out = sys.argv[1], sys.argv[2]
queries = {}
for d in sorted(glob.glob(os.path.join(sdir, "*"))):
    if not os.path.isdir(d): continue
    qid = os.path.basename(d)               # e.g. 5sdy  (dot-free PDB IDs: OK, issue #176)
    seqs = []
    for fa in glob.glob(os.path.join(d, "*.fasta")):
        cur=[]
        for line in open(fa):
            if line.startswith(">"):
                if cur: seqs.append("".join(cur)); cur=[]
            else: cur.append(line.strip())
        if cur: seqs.append("".join(cur))
    chains=[{"molecule_type":"protein","chain_ids":[chr(65+i)],"sequence":s}
            for i,s in enumerate(dict.fromkeys(seqs)) if s]   # unique seqs
    if chains: queries[qid] = {"chains": chains}
json.dump({"queries": queries}, open(out,"w"), indent=2)
print(f"wrote {out} with {len(queries)} queries")
PY
  # 2) Ask the ColabFold server for alignments (writes per-chain a3m + query_msa.json).
  #    SCOPE: ColabFold is used for MSAs ONLY. `align-msa-server` calls
  #    preprocess_colabfold_msas(), which defaults use_templates=False, so NO
  #    templates and NO structures are fetched from ColabFold. (verified in source)
  CF_OUT="$WORK/colabfold_raw"; mkdir -p "$CF_OUT"
  echo "    contacting ColabFold server for MSAs only (no templates, no databases)..."
  run_openfold align-msa-server --query-json "$CF_QUERY" --output-dir "$CF_OUT"
  # Guard: align-msa-server must not have produced template artifacts.
  if find "$CF_OUT" -iname "*template*" -o -iname "*.pdb" 2>/dev/null | grep -q .; then
    echo "    WARNING: unexpected template/structure files under $CF_OUT — ColabFold should fetch MSAs only."
  fi
  # 3) BRIDGE: map ColabFold filenames -> the names the OF3 trainer expects, into
  #    per-chain folders named <query>_<chain> under $MSA_RAW.
  python3 - "$CF_OUT" "$MSA_RAW" <<'PY'
import sys, os, glob, shutil
src, dst = sys.argv[1], sys.argv[2]
# ColabFold per-chain outputs -> training-expected filenames (dataset_config_components.py)
MAP = {
    "uniref.a3m": "uniref90_hits.a3m",
    "bfd.mgnify30.metaeuk30.smag30.a3m": "bfd_uniref_hits.a3m",
    "pair.a3m": "uniprot_hits.a3m",   # paired source; trainer pairs from uniprot_hits
}
os.makedirs(dst, exist_ok=True)
made = 0
# ColabFold writes one subdir per chain; find dirs that contain any a3m we know
for d in glob.glob(os.path.join(src, "**"), recursive=True):
    if not os.path.isdir(d): continue
    a3ms = {os.path.basename(f): f for f in glob.glob(os.path.join(d, "*.a3m"))}
    if not any(k in a3ms for k in MAP): continue
    chain_name = os.path.basename(d.rstrip("/"))
    outdir = os.path.join(dst, chain_name); os.makedirs(outdir, exist_ok=True)
    for cf_name, of3_name in MAP.items():
        if cf_name in a3ms:
            shutil.copy(a3ms[cf_name], os.path.join(outdir, of3_name))
    made += 1
print(f"bridged {made} chain MSA folder(s) into {dst}")
if made == 0:
    print("WARNING: no ColabFold a3m files found to bridge — inspect", src)
PY
  echo "    NOTE: ColabFold MSAs approximate (not identical to) the OF3 training MSA recipe."
  echo "          Fine for small fine-tunes/smoke tests; use snakemake DBs for production parity."

elif [[ "$MSA_MODE" == "snakemake" ]]; then
  echo "    Collecting FASTAs + writing MSA config"
  # Each preprocessed structure has a .fasta next to it; collect query sequences:
  ALL_FASTA="$WORK/all_queries.fasta"
  python "$SCRIPTS/data_preprocessing/collect_preprocessed_fastas.py" \
      --preprocessed-dir "$STRUCT_DIR" --output "$ALL_FASTA" 2>/dev/null || \
      cat "$STRUCT_DIR"/*/*.fasta > "$ALL_FASTA"   # fallback: simple concat

  cat > "$WORK/msa_config.json" <<JSON
{
  "input_fasta": "$ALL_FASTA",
  "openfold_env": "$CONDA_PREFIX",
  "databases": ["uniref90", "uniprot", "mgnify", "bfd"],
  "base_database_path": "$DB_PATH",
  "output_directory": "$MSA_RAW",
  "jackhmmer_output_format": "sto",
  "jackhmmer_threads": 4,
  "hhblits_threads": 16,
  "tmpdir": "/tmp",
  "run_template_search": false
}
JSON
  echo "    Running snakemake MSA pipeline (needs DBs at $DB_PATH)"
  echo "    If DBs are missing: python $SCRIPTS/snakemake_msa/download_of3_databases.py download --output-dir $DB_PATH --jackhmmer-dbs uniref90,uniprot,mgnify --download-bfd"
  snakemake -s "$SCRIPTS/snakemake_msa/MSA_Snakefile" \
            --configfile "$WORK/msa_config.json" --cores "$NWORKERS"
else
  echo "    MSA_MODE=skip -> expecting precomputed per-chain MSA folders in $MSA_RAW"
  echo "    Each folder named <pdbid>_<chain> with files like uniref90_hits.sto, uniprot_hits.sto, ..."
fi

echo "    Building the MSA representatives FASTA"
# db stems to read the query sequence from (must match the per-chain filenames present)
if [[ "$MSA_MODE" == "colabfold" ]]; then
  REP_DBS="uniref90_hits,uniprot_hits,bfd_uniref_hits"
else
  REP_DBS="uniref90,uniprot,mgnify,bfd"
fi
python "$SCRIPTS/utils/generate_representatives_from_msa_directory.py" \
  --msa-directory "$MSA_RAW" \
  --out-fasta "$REPS_FASTA" \
  --protein-dbs "$REP_DBS" \
  --ncores "$NWORKERS"

echo "    Pre-parsing MSAs into npz (REQUIRED by the trainer)"
python "$SCRIPTS/data_preprocessing/preparse_alignments_of3.py" \
  --alignments_directory "$MSA_RAW" \
  --alignment_array_directory "$ALN_ARR" \
  --max_seq_counts '{"uniref90_hits":10000,"uniprot_hits":50000,"mgnify_hits":5000,"bfd_uniref_hits":5000}' \
  --num_workers "$NWORKERS"

echo "==> [5/6] Training dataset cache (the FINAL trainer input)"
python "$SCRIPTS/data_preprocessing/create_pdb-weighted_training_dataset_cache.py" \
  --metadata-cache "$META_CACHE" \
  --preprocessed-dir "$STRUCT_DIR" \
  --alignment-representatives-fasta "$REPS_FASTA" \
  --output "$CACHE_DIR/training_cache.json" \
  --dataset-name "PDB-weighted" \
  --max-resolution "$TRAIN_MAX_RES" \
  --allow-missing-alignment \
  --missing-alignment-log "$CACHE_DIR/train_missing_alignments.log" \
  --log-level INFO

echo "==> [6/6] Validation dataset cache (held-out, homology-filtered vs. train)"
python "$SCRIPTS/data_preprocessing/create_pdb_validation_dataset_cache.py" \
  --metadata-cache "$META_CACHE" \
  --preprocessed-dir "$STRUCT_DIR" \
  --train-dataset-cache "$CACHE_DIR/training_cache.json" \
  --alignment-representatives-fasta "$REPS_FASTA" \
  --output "$CACHE_DIR/validation_cache.json" \
  --dataset-name "PDB-validation" \
  --max-resolution "$VAL_MAX_RES" \
  --min-release-date "$VAL_MIN_RELEASE_DATE" \
  --allow-missing-alignment \
  --missing-alignment-log "$CACHE_DIR/val_missing_alignments.log" \
  --log-level INFO

cat <<EOF

================================================================================
DONE. Point your training YAML 'dataset_paths' at:
  alignment_array_directory:    $ALN_ARR
  dataset_cache_file (train):   $CACHE_DIR/training_cache.json
  dataset_cache_file (val):     $CACHE_DIR/validation_cache.json
  target_structures_directory:  $STRUCT_DIR
  reference_molecule_directory: $REF_MOLS   (verify this folder name)
Then launch:
  run_openfold train --runner-yaml finetune_lowN_single_gpu.yml --seed 42
================================================================================
EOF
