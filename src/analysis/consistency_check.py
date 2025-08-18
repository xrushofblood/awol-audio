import os, csv, argparse, yaml
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    raw_dir = Path(cfg["paths"]["raw_dir"])
    proc_dir = Path(cfg["paths"]["proc_dir"])
    meta_csv = Path(cfg["paths"]["meta_csv"])

    raw = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(".wav")])
    proc = sorted([f for f in os.listdir(proc_dir) if f.lower().endswith(".wav")])

    with open(meta_csv, "r", encoding="utf-8") as f:
        names = [r["filename"] for r in csv.DictReader(f)]

    raw_only = [f for f in raw if f not in names]
    csv_only = [f for f in names if f not in raw]
    missing_proc = [f for f in names if f not in proc]

    print(f"[INFO] raw: {len(raw)} | processed: {len(proc)} | csv rows: {len(names)}")
    if raw_only:
        print(f"[WARN] Files in raw/ not listed in CSV: {raw_only[:5]} ... ({len(raw_only)})")
    if csv_only:
        print(f"[WARN] Files listed in CSV but missing in raw/: {csv_only[:5]} ... ({len(csv_only)})")
    if missing_proc:
        print(f"[WARN] Files listed in CSV but missing in processed/: {missing_proc[:5]} ... ({len(missing_proc)})")
    if not (raw_only or csv_only or missing_proc):
        print("[OK] CSV, raw/, and processed/ are consistent.")

if __name__ == "__main__":
    main()
