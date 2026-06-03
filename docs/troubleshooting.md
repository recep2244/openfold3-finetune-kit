# Troubleshooting

The failure modes you are most likely to hit, with verified fixes.

## `setup_openfold` exits instantly: *"workspace requires pixi '>=0.68'"*
Your pixi is too old (a fresh install can be 0.63). Fix:

```bash
pixi self-update
```

## `Unable to JIT load evoformer_attn` / `evoformer_attn.so: cannot open shared object file`
The DeepSpeed evoformer kernel failed to compile.

=== "pixi install (this kit's default)"
    **Do not set `CUDA_HOME`.** The pixi environment ships its own matched `nvcc` and cutlass;
    pointing it at a different system CUDA (e.g. system 12.0 vs the env's 12.9) breaks the build.

    ```bash
    unset CUDA_HOME
    rm -rf ~/openfold-3/.pixi/envs/openfold3-cuda12/caches/torch_extensions/evoformer_attn
    ```

=== "pip / conda install"
    Here you *do* need the toolkit on the path:

    ```bash
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    export CUTLASS_PATH=$(python -c "import cutlass_library,pathlib;print(pathlib.Path(cutlass_library.__file__).resolve().parent/'source')")
    ```

## Prediction "succeeded" but produced no structure
`run_openfold` can exit `0` even when a query fails (it reports
`Successful Queries: 0 / Failed Queries: 1`). Always confirm a `.cif` was written, and read
the real error:

```bash
cat <output-dir>/logs/predict_err_rank0.log
```

`verify_setup.sh` already checks for an actual structure file rather than trusting the exit code.

## CUDA out of memory during training
Your GPU is too small for the selected profile. Use `GPU_PROFILE="small"`, lower
`token_budget` in the config, or move to a larger GPU (see [Cloud GPU](cloud.md)).

## MSA / ColabFold step is slow or fails
The public server can be busy. Wait and re-run — completed steps are cached. For
private/sensitive sequences, do not use the public server; use local databases
(`MSA_MODE=snakemake`, advanced).

## `DataLoader worker exited`
A known issue, already worked around (single worker). If it persists, the machine is low on RAM.

## Query name error
Do not put dots (`.`) in structure or query names — a known OpenFold3 limitation.

## Data prep fails: `No module named 'Bio'` or `mmseqs: not found` (exit 127)
The data-prep / dataset-cache scripts in `openfold-3` need two tools that the
`openfold3-cuda12` environment does not always include:

```bash
# inside the env (pixi shell -e openfold3-cuda12, or via `pixi run`)
pip install biopython                                  # MSA representatives step
conda install -c conda-forge -c bioconda mmseqs2       # or: download the mmseqs static binary onto PATH
```

`biopython` is needed by `generate_representatives_from_msa_directory.py`; **MMseqs2** is needed
for sequence clustering in the training-dataset cache. Install both before running `run_all.sh`.

## "Invalid max_seq_counts … bfd_uniref_hits Extra inputs are not permitted"
A version-drift symptom: older OpenFold3 used `bfd_uniref_hits`; current `main` renamed the
HHblits key to `bfd_hits`. This kit targets current `main` (`bfd_hits`); if you pin an older
OpenFold3, revert the `bfd_hits` name in `scripts/prepare_data.sh`.

## Evaluation scoring errors
Usually a Docker/OpenStructure issue. Training and prediction still succeeded; you can score
later once the image is available.
