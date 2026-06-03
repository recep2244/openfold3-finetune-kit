#!/usr/bin/env python3
"""Composite QC gate and ranker for OpenFold3 predictions.

Reads a CSV of per-prediction confidence metrics, applies research-backed
threshold filters, computes a composite score, and writes a ranked CSV.

Why composite: individual confidence metrics have only modest, dataset-dependent
correlation with experimental success — they work as *pre-screening filters*, not as
affinity predictors. A weighted composite tends to rank better than any single metric.
Thresholds are pre-screening heuristics (several borrowed from binder-design QC); ipSAE
follows Dunbrack (2025), bioRxiv 2025.02.10.637595.

Recognised columns (any subset, case-insensitive): id, plddt, ptm, iptm,
pae_interaction, ipsae, esm2_pll. pLDDT on a 0-100 scale is auto-normalised to 0-1.

Usage:
  python scripts/qc_gate.py metrics.csv --out ranked.csv
  python scripts/qc_gate.py metrics.csv --stringent
  python scripts/qc_gate.py --self-test
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from pathlib import Path

# (key, standard, stringent, comparison) — comparison: ">=" higher-is-better, "<=" lower-is-better
THRESHOLDS = {
    "plddt": (0.85, 0.90, ">="),
    "ptm": (0.70, 0.80, ">="),
    "iptm": (0.50, 0.60, ">="),
    "pae_interaction": (12.0, 10.0, "<="),
    "ipsae": (0.61, 0.70, ">="),
}
# Composite weights (renormalised over the metrics actually present).
WEIGHTS = {"plddt": 0.30, "ipsae": 0.25, "iptm": 0.20, "pae_interaction": 0.15, "esm2_pll": 0.10}


def _norm(key: str, value: float) -> float:
    """Map a metric onto 0-1 where higher is better."""
    if key == "plddt":
        return value / 100.0 if value > 1.5 else value
    if key == "pae_interaction":
        return max(0.0, 1.0 - value / 20.0)  # 0 A -> 1.0, 20 A -> 0.0
    return value


def composite(row: dict[str, float]) -> float:
    present = {k: w for k, w in WEIGHTS.items() if k in row}
    if not present:
        return 0.0
    total = sum(present.values())
    return sum(w / total * _norm(k, row[k]) for k, w in present.items())


def passes(row: dict[str, float], stringent: bool) -> bool:
    for key, (std, stri, cmp) in THRESHOLDS.items():
        if key not in row:
            continue
        thr = stri if stringent else std
        v = row[key] / 100.0 if key == "plddt" and row[key] > 1.5 else row[key]
        if cmp == ">=" and v < thr:
            return False
        if cmp == "<=" and v > thr:
            return False
    return True


def load(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        keymap = {k: (k.strip().lower()) for k in (reader.fieldnames or [])}
        for raw in reader:
            row: dict[str, float | str] = {}
            for k, v in raw.items():
                lk = keymap[k]
                if lk == "id":
                    row["id"] = v
                else:
                    with contextlib.suppress(TypeError, ValueError):
                        row[lk] = float(v)
            rows.append(row)
    return rows


def rank(rows: list[dict], stringent: bool) -> list[dict]:
    out = []
    for r in rows:
        metrics = {k: v for k, v in r.items() if isinstance(v, float)}
        r = dict(r)
        r["passed"] = passes(metrics, stringent)
        r["composite"] = round(composite(metrics), 4)
        out.append(r)
    out.sort(key=lambda r: (r["passed"], r["composite"]), reverse=True)
    return out


def report(ranked: list[dict]) -> None:
    n = len(ranked)
    npass = sum(1 for r in ranked if r["passed"])
    rate = (npass / n * 100) if n else 0.0
    print(f"predictions: {n}    passed gate: {npass} ({rate:.0f}%)")
    status = (
        "excellent" if rate > 15 else "good" if rate >= 10 else "marginal" if rate >= 5 else "poor"
    )
    print(f"campaign health: {status}")
    print("\ntop by composite score:")
    print(f"  {'id':<20} {'composite':>9}  {'pass':>5}")
    for r in ranked[:10]:
        print(f"  {str(r.get('id', '?')):<20} {r['composite']:>9.4f}  {str(r['passed']):>5}")


def write(ranked: list[dict], out: Path) -> None:
    cols = ["id"] + sorted({k for r in ranked for k in r if k not in ("id",)})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ranked)
    print(f"\nwrote ranked CSV -> {out}")


def self_test() -> int:
    rows = [
        {"id": "good", "plddt": 91.0, "iptm": 0.72, "pae_interaction": 8.0, "ipsae": 0.74},
        {"id": "mid", "plddt": 0.86, "iptm": 0.55, "pae_interaction": 11.0, "ipsae": 0.62},
        {"id": "fail", "plddt": 70.0, "iptm": 0.30, "pae_interaction": 18.0, "ipsae": 0.40},
    ]
    ranked = rank(rows, stringent=False)
    assert ranked[0]["id"] == "good", ranked
    assert ranked[-1]["id"] == "fail", ranked
    assert ranked[0]["passed"] and not ranked[-1]["passed"]
    assert ranked[0]["composite"] > ranked[1]["composite"] > ranked[2]["composite"]
    print("self-test OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("metrics_csv", nargs="?", help="CSV of per-prediction metrics")
    p.add_argument("--out", default="qc_ranked.csv", help="output ranked CSV")
    p.add_argument("--stringent", action="store_true", help="use stringent thresholds")
    p.add_argument("--self-test", action="store_true", help="run a built-in sanity test and exit")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if not args.metrics_csv:
        p.error("metrics_csv is required (or pass --self-test)")

    rows = load(Path(args.metrics_csv))
    if not rows:
        sys.exit("no rows parsed from input CSV")
    ranked = rank(rows, args.stringent)
    report(ranked)
    write(ranked, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
