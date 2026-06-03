# Contributing

Thanks for your interest in improving **openfold3-finetune-kit**.

## Ground rules
- This kit drives the upstream [OpenFold3](https://github.com/aqlaboratory/openfold-3)
  package; it does not vendor model code. Keep changes in the kit's scripts,
  configs, docs, notebooks, Docker, and Hugging Face assets.
- Shell scripts target `bash` and must pass `shellcheck` and `bash -n`.
- YAML configs must stay valid (`yamllint`) and mirror keys that exist in the
  OpenFold3 source. Cite the source path in a comment when you add a key.
- Notebooks must be saved with cleared outputs (`.ipynb_checkpoints` is ignored).

## Local checks before opening a PR
Install the hooks once, then let them run on every commit:
```bash
pip install pre-commit && pre-commit install
```

Run the full lint suite (shellcheck + yamllint + ruff + notebook validation) and, if you have
a GPU with OpenFold3 installed, the end-to-end readiness check:
```bash
make lint
make verify        # needs a GPU + OpenFold3 installed
```

Preview the documentation site locally with `make docs`.

## Pull requests
- One focused change per PR. Describe what you changed and how you tested it.
- Do not commit model weights, datasets, or run outputs (see `.gitignore`).
- By contributing you agree your changes are licensed under Apache-2.0.
