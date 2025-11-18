#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Add/overwrite tgt_units column in TSV manifest.")
    p.add_argument("--in", dest="in_path", required=True, help="Input TSV manifest")
    p.add_argument("--out", dest="out_path", required=True, help="Output TSV manifest")
    p.add_argument("--units-root", required=True, help="Base directory for unit files")
    p.add_argument("--suffix", default=".npy", help="Units file suffix (e.g., .npy, .npz, .txt)")
    p.add_argument("--pattern", default="{id}", help="Filename pattern using {id}")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    units_root = Path(args.units_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        if "tgt_units" not in fieldnames:
            fieldnames.append("tgt_units")
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if not row or row.get("id") == "id":
                continue
            sample_id = row["id"]
            fname = args.pattern.format(id=sample_id) + args.suffix
            units_path = units_root / fname
            row["tgt_units"] = str(units_path)
            writer.writerow(row)

if __name__ == "__main__":
    main()





