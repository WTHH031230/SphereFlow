#!/usr/bin/env python3
"""Materialize the released train/val/test splits for the local machine.

The files in ``data/splits/`` store slice paths relative to the processed-data
root (``brats/<phase>/<case>_<slice>.npy`` and ``mri_ct/<phase>/<file>.npy``),
i.e. the layout written by ``preprocess_brats.py`` and
``preprocess_mri_ct.py`` under ``$DATA_ROOT``. All training / evaluation
code reads split JSONs with absolute ``path`` entries, so this script prefixes
every entry with ``--data-root`` and writes the result to ``--out-dir``.
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAMES = ["brats2023_distribution.json", "brain_mri_ct_distribution.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Processed data root containing brats/ and/or mri_ct/.")
    parser.add_argument("--splits-dir", type=Path, default=REPO / "data" / "splits")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data")
    parser.add_argument("--check", action="store_true", help="Fail if a referenced .npy file is missing.")
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        src = args.splits_dir / name
        if not src.is_file():
            print(f"skip {name}: {src} not found")
            continue
        records = json.loads(src.read_text())
        dataset_dir = records["train"][0]["path"].split("/")[0]
        if not (root / dataset_dir).is_dir():
            print(f"skip {name}: {root / dataset_dir} does not exist")
            continue
        missing = 0
        for split in records.values():
            for rec in split:
                rec["path"] = os.path.join(str(root), rec["path"])
                if args.check and not os.path.isfile(rec["path"]):
                    missing += 1
        if missing:
            raise SystemExit(f"{name}: {missing} referenced files are missing under {root}")
        out = args.out_dir / name
        out.write_text(json.dumps(records, indent=1) + "\n")
        sizes = ", ".join(f"{k}={len(v)}" for k, v in records.items())
        print(f"wrote {out} ({sizes})")


if __name__ == "__main__":
    main()
