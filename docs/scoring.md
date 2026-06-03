# Scoring, QC & homology

Beyond the baseline-vs-fine-tuned comparison in [evaluation](pipeline.md#d-evaluation),
the kit ships three research-backed tools for triaging predictions and curating targets.

## ipSAE — reference-free interface confidence

`evaluate.sh` needs an experimental structure. **ipSAE** (Dunbrack et al., 2025) instead
scores an interface from the model's own PAE — no reference required — so you can rank
fine-tuned predictions on targets you cannot yet validate. It outperforms ipTM for ranking
(≈1.4× precision), especially across different-length constructs and disordered regions.

```bash
# OpenFold3 / AF3-style outputs: the *_full_data*.json (PAE) + the predicted .cif
bash scripts/ipsae_score.sh out/target/*full_data_0.json out/target/*_model.cif 10 10
```

| Metric | Standard | Stringent |
|---|---|---|
| ipSAE_min | > 0.61 | > 0.70 |
| LIS | > 0.35 | > 0.45 |
| pDockQ | > 0.50 | > 0.60 |

The helper fetches [DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE) into `~/.cache/IPSAE`
on first run. Paper: *"What's wrong with AlphaFold's ipTM score"* (bioRxiv, 2025).

## QC gate & composite ranker

Individual confidence metrics are weak predictors of experimental success (single-metric
ROC AUC ≈ 0.65) — they are **pre-screening filters, not affinity predictors**. `qc_gate.py`
applies threshold filters and a renormalised **composite** score (which ranks better than any
single metric), then writes a ranked CSV and a campaign-health summary.

```bash
# metrics.csv columns (any subset): id, plddt, ptm, iptm, pae_interaction, ipsae, esm2_pll
python scripts/qc_gate.py metrics.csv --out qc_ranked.csv          # standard thresholds
python scripts/qc_gate.py metrics.csv --stringent                  # tighter gate
python scripts/qc_gate.py --self-test                              # sanity check
```

Thresholds follow [Configuration → interpreting scores](pipeline.md#interpreting-the-scores)
(pLDDT > 0.85, ipTM > 0.50, PAE-interface < 12 Å, ipSAE > 0.61). pLDDT given 0–100 is
auto-normalised. Campaign health: > 15 % pass = excellent, 10–15 % good, 5–10 % marginal, < 5 % poor.

## Foldseek — structural homologs & novelty

Use structure search to **curate** train/held-out sets (find homologs of your target) and to
**gauge novelty** — how far a target sits from known structures is exactly where the base model
is weakest and fine-tuning helps most.

```bash
# set up a database once:  foldseek databases PDB pdb100 tmp/
bash scripts/foldseek_search.sh target.cif pdb100 hits.m8
# top-hit identity < 30%  ->  novel fold
```

Needs [Foldseek](https://github.com/steineggerlab/foldseek)
(`conda install -c conda-forge -c bioconda foldseek`); `afdb50` covers ~67M structures for
remote-homolog detection.

## Beyond fine-tuning

These are out of scope for this kit (which adapts an existing model), but are natural
next steps once you have a tuned checkpoint:

- **De novo binder design** — RFdiffusion, BindCraft, BoltzGen, ProteinMPNN/SolubleMPNN.
- **Pipeline productionization** — wrapping the stages in a workflow engine (e.g. Nextflow/Snakemake) for large campaigns.

The QC gate, ipSAE, and Foldseek helpers above transfer directly to those workflows.
