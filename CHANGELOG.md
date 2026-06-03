# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-03
### Added
- End-to-end fine-tuning pipeline (`scripts/run_all.sh`): data prep, preflight, training, evaluation.
- Setup verifier (`scripts/verify_setup.sh`) that runs a real smoke prediction.
- Fine-tune configurations for single-GPU, multi-GPU, anti-forgetting, and 12 GB smoke test.
- Colab notebooks for fine-tuning and inference.
- Docker image and compose setup.
- Hugging Face Gradio Space, model card, and checkpoint upload script.
- MkDocs-Material documentation site and GitHub Pages deployment.
- Developer tooling: Makefile, pre-commit, ruff, EditorConfig; CI for lint and docs.

[Unreleased]: https://github.com/recep2244/openfold3-finetune-kit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/recep2244/openfold3-finetune-kit/releases/tag/v0.1.0
