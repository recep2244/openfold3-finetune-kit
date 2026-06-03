# Security Policy

## Supported versions
This project is pre-1.0; only the latest `main` is supported.

## Reporting a vulnerability
Please report security issues privately via GitHub's
[private vulnerability reporting](https://github.com/recep2244/openfold3-finetune-kit/security/advisories/new)
rather than opening a public issue. Include reproduction steps and the affected version/commit.
You can expect an initial response within a few days.

## Scope and notes
- This kit downloads model weights and fetches MSAs from the public ColabFold server. Do **not**
  submit proprietary or sensitive sequences to the public server; use local MSA databases
  (`MSA_MODE=snakemake`) instead.
- Never commit credentials or tokens. The Hugging Face upload script reads `HF_TOKEN` from the
  environment and must not be hardcoded.
