#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path


def guess_data_root(manifest_path: Path) -> Path:
    # Prefer project root/data if present, else manifest's parent
    prj = manifest_path.resolve().parents[1]
    data_dir = prj / "data"
    return data_dir if data_dir.exists() else manifest_path.parent


def ensure_wav_path(data_root: Path, value: str) -> str:
    # If already absolute and exists, return
    p = Path(value)
    if p.is_absolute() and p.exists():
        return str(p)
    # Common relative roots
    candidates = [
        data_root / value,
        data_root / "wavs" / value,
        data_root / value.lstrip("/"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str(candidates[0])  # fallback (may not exist)


def ensure_npy_path(data_root: Path, src_audio_abs: str, units_root: Path) -> str:
    # Construct units path using basename + .npy under units_root
    base = Path(src_audio_abs).stem
    units_root.mkdir(parents=True, exist_ok=True)
    return str((units_root / f"{base}.npy").resolve())


def fix_manifest(manifest_path: Path, out_path: Path, units_root: Path, check_only: bool = False) -> int:
    data_root = guess_data_root(manifest_path)
    fixed = 0
    with manifest_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        fieldnames = reader.fieldnames or []
        # Ensure required columns
        required = ["id", "src_audio", "src_text", "tgt_audio", "tgt_text"]
        for col in required:
            if col not in fieldnames:
                fieldnames.append(col)
        # Add tgt_units column if missing
        if "tgt_units" not in fieldnames:
            fieldnames.append("tgt_units")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            try:
                src_audio = row.get("src_audio", "").strip()
                if src_audio.endswith(".npy"):
                    # Bad: src_audio should be wav. Try to recover using id or tgt_audio
                    source_hint = row.get("tgt_audio", row.get("id", ""))
                    src_audio = source_hint if source_hint else src_audio
                src_audio_abs = ensure_wav_path(data_root, src_audio)
                tgt_units = row.get("tgt_units", "").strip()
                if not tgt_units.endswith(".npy"):
                    tgt_units = ensure_npy_path(data_root, src_audio_abs, units_root)
                # Record fixes
                if row.get("src_audio", "").strip() != src_audio or row.get("tgt_units", "").strip() != tgt_units:
                    fixed += 1
                row["src_audio"] = src_audio_abs
                row["tgt_units"] = tgt_units
                writer.writerow(row)
            except Exception:
                # If any row fails, write it as-is to not lose data
                writer.writerow(row)
    return fixed


def main():
    ap = argparse.ArgumentParser(description="Fix *.units.tsv so that src_audio points to WAV and tgt_units points to .npy in units_root.")
    ap.add_argument("--manifest", required=True, help="Input *.units.tsv")
    ap.add_argument("--out", required=False, help="Output path (default: overwrite *.fixed.tsv next to input)")
    ap.add_argument("--units-root", required=True, help="Directory with unit .npy files")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_path = Path(args.out) if args.out else manifest.with_suffix(".fixed.tsv")
    units_root = Path(args.units_root)

    if not manifest.exists():
        print(f"Manifest not found: {manifest}", file=sys.stderr)
        sys.exit(1)

    fixed = fix_manifest(manifest, out_path, units_root)
    print(f"Wrote: {out_path} (rows fixed: {fixed})")


if __name__ == "__main__":
    main()


