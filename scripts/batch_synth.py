# scripts/batch_synth.py
import sys
import subprocess, os, csv, time

CONFIG = "configs/synth.yaml"
SEED = 0
OUTCSV = "outputs/gen_audio/batch_run_log.csv"

PROMPTS = [
  "low-pitch dark mellow pluck with short decay",
  "low-pitch warm pluck with medium decay",
  "low-pitch bright metallic pluck with long sustain",
  "dark mellow pluck with long decay",
  "warm wooden pluck with short decay",
  "bright metallic pluck with medium decay",
  "high-pitch dark mellow pluck with medium decay",
  "high-pitch warm pluck with long sustain",
  "high-pitch very bright metallic pluck with very short decay",
]

os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)

rows = []
for p in PROMPTS:
    cmd = [
        "python", "-m", "src.synth.run_synth",
        "--config", CONFIG,
        "--prompt", p,
        "--seed", str(SEED),
    ]
    print(">>", " ".join(cmd))
    t0 = time.time()
    python_exec = sys.executable  
    result = subprocess.run([python_exec, "src/synth/run_synth.py", "--prompt", p])
    dt = time.time() - t0
    ok = result.returncode == 0
    print(result.stdout)
    if not ok:
        print("ERR:", result.stderr)

    rows.append({
        "prompt": p,
        "ok": ok,
        "seconds": f"{dt:.2f}",
        "stdout_tail": "\n".join(result.stdout.strip().splitlines()[-5:]) if result.stdout else "",
        "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-5:]) if result.stderr else "",
    })

with open(OUTCSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"[BATCH] Done. Log saved to {OUTCSV}")
