# src/pipeline/batch_text2synth.py
import csv, argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline_real.yaml")
    ap.add_argument("--csv", required=True)  # expects column "prompt"
    ap.add_argument("--no-neigh", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    for r in rows:
        p = r.get("prompt","").strip()
        if not p: 
            continue
        cmd = [
            sys.executable, "-m", "src.pipeline.text2synth",
            "--config", args.config, "--query", p
        ]
        if args.no_neigh:
            cmd += ["topk", "0"]
        print(">>", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
