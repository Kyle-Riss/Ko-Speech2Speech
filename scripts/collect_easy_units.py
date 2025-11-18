#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import shutil
import re


EASY_PHRASES_DEFAULT = [
    "hello.",
    "hello",
    "thank you.",
    "thank you",
    "yes.",
    "yes",
    "no.",
    "no",
    "okay.",
    "okay",
    "good morning.",
    "good morning",
    "good evening.",
    "good evening",
    "good night.",
    "good night",
]


def norm(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def iter_units_records(tsv_path: Path):
    """
    Yields tuples: (row_dict, units_path_str)
    Handles two-line per sample format seen in *sampled.units.tsv:
      line1: id, src_audio, src_text, tgt_audio, tgt_text
      line2: /abs/path/to/units.npy
    Also handles single-line format with tgt_units column if present.
    """
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return
        # Detect header with tgt_units
        has_tgt_units = "tgt_units" in header
        if has_tgt_units:
            idx_id = header.index("id")
            idx_tgt_text = header.index("tgt_text")
            idx_units = header.index("tgt_units")
            for row in reader:
                if not row or row[0] == "id":
                    continue
                yield (
                    {
                        "id": row[idx_id],
                        "tgt_text": row[idx_tgt_text],
                    },
                    row[idx_units],
                )
        else:
            # Two-line format: header line, then a line with 'tgt_units' label
            # After that, pairs of (meta line, units path line)
            # We will read raw lines to pair them robustly.
            f.seek(0)
            lines = [ln.rstrip("\n") for ln in f]
            # Skip first 2 header lines if match expected pattern
            start_idx = 0
            if lines and lines[0].startswith("id\t"):
                start_idx = 1
            if start_idx < len(lines) and lines[start_idx].strip() == "tgt_units":
                start_idx += 1
            # Now from start_idx, iterate pairs
            for i in range(start_idx, len(lines), 2):
                if i + 1 >= len(lines):
                    break
                meta = lines[i]
                units_path = lines[i + 1].strip()
                parts = meta.split("\t")
                if len(parts) < 5:
                    continue
                row = {
                    "id": parts[0],
                    "tgt_text": parts[4],
                }
                yield row, units_path


def main():
    ap = argparse.ArgumentParser(description="Collect units for simple phrases into a target directory.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/test_sampled.units.tsv",
            "data/dev_sampled.units.tsv",
            "data/train_sampled.units.tsv",
        ],
        help="List of *sampled.units.tsv files to scan (ordered by preference).",
    )
    ap.add_argument(
        "--phrases",
        nargs="*",
        default=EASY_PHRASES_DEFAULT,
        help="Target phrases to collect (case-insensitive).",
    )
    ap.add_argument(
        "--outdir",
        default="forced_units",
        help="Output directory to place copied units as <phrase>.npy",
    )
    args = ap.parse_args()

    phrases = [norm(p) for p in (args.phrases or [])]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    found = {}
    for tsv in args.inputs:
        tsv_path = Path(tsv)
        if not tsv_path.exists():
            continue
        for row, units_path in iter_units_records(tsv_path):
            tgt = norm(row.get("tgt_text", ""))
            if tgt in phrases and tgt not in found:
                try:
                    src = Path(units_path)
                    if not src.exists():
                        continue
                    # filename: phrase with underscores, no punctuation
                    fname = re.sub(r"[^a-z0-9]+", "_", tgt).strip("_") + ".npy"
                    dst = outdir / fname
                    shutil.copyfile(src, dst)
                    found[tgt] = str(dst)
                except Exception:
                    pass
        # stop early if all phrases found
        if len(found) == len(phrases):
            break

    print("Collected units:")
    for p in phrases:
        print(f"- {p}: {found.get(p, 'NOT FOUND')}")
    print(f"\nOutput directory: {outdir.resolve()}")
    if not found:
        print("No phrases found. Consider extending --inputs or --phrases.")


if __name__ == "__main__":
    main()



