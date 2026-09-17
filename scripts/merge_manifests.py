"""Merge square manifests (e.g. synthetic + real) into one shuffled manifest.
Usage: python scripts/merge_manifests.py data/synth_squares/manifest.csv \
           data/squares_manifest.csv -o data/combined_manifest.csv
"""
import argparse
import csv
import random


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("-o", "--out", default="data/combined_manifest.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rows = []
    for m in args.manifests:
        with open(m) as f:
            rows += list(csv.DictReader(f))
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "photo_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(rows)} squares from {len(args.manifests)} manifests -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
