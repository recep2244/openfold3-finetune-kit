#!/usr/bin/env python3
"""
Publish a fine-tuned OpenFold3 checkpoint to the Hugging Face Hub.

Creates (or reuses) a model repo and uploads the checkpoint plus the model card.
Run it yourself when ready — it is NOT executed by the kit automatically.

Auth: `huggingface-cli login` once, or set HF_TOKEN in the environment.

Examples
--------
  python huggingface/upload_to_hub.py \
      --ckpt ~/openfold3_run/train_out/checkpoints/last.ckpt \
      --repo-id recep2244/openfold3-pde10a

  python huggingface/upload_to_hub.py --ckpt last.ckpt --repo-id me/of3-mytarget --private
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Path to the fine-tuned .ckpt/.pt")
    p.add_argument("--repo-id", required=True, help="HF repo id, e.g. user/openfold3-mytarget")
    p.add_argument("--model-card", default=str(Path(__file__).with_name("model_card.md")))
    p.add_argument("--private", action="store_true", help="Create a private repo")
    p.add_argument(
        "--token", default=os.environ.get("HF_TOKEN"), help="HF token (or use HF_TOKEN / hf login)"
    )
    args = p.parse_args()

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_file():
        sys.exit(f"checkpoint not found: {ckpt}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Install the client first:  pip install huggingface_hub")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"-> repo ready: {args.repo_id}")

    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=ckpt.name,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"-> uploaded {ckpt.name}")

    card = Path(args.model_card)
    if card.is_file():
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("-> uploaded model card as README.md (edit the <TARGET>/metrics placeholders)")

    print(f"\nDone: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
