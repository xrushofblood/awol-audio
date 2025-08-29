# src/pipeline/collect_text2synth_csv.py
import argparse, csv, json, re
from pathlib import Path

PARAM_KEYS = ["pitch_hz", "decay_t60", "brightness", "damping", "pick_position", "noise_mix"]

def slugify(s: str) -> str:
    """Convert prompt to a simple slug that likely appears in output filenames."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def load_prompts_map(prompts_csv: Path):
    """
    Read an optional prompts CSV. Accepts column name 'query' or 'prompt'.
    Returns: dict {slug -> original_prompt}
    """
    if not prompts_csv:
        return {}
    with open(prompts_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # find header
    col = None
    if rows:
        cols = [c.lower() for c in rows[0].keys()]
        if "query" in cols: col = [k for k in rows[0].keys() if k.lower()=="query"][0]
        elif "prompt" in cols: col = [k for k in rows[0].keys() if k.lower()=="prompt"][0]
    if not col:
        print(f"[WARN] No 'query' or 'prompt' column found in {prompts_csv}. Will not map prompts.")
        return {}
    mp = {}
    for r in rows:
        p = (r.get(col) or "").strip()
        if p:
            mp[slugify(p)] = p
    return mp

def extract_params(d: dict):
    """
    Try a few shapes:
    - flat keys at top-level
    - nested under d['params']
    - nested under d['prediction'] or similar
    Returns dict with PARAM_KEYS -> float (or None if missing).
    """
    # Candidates to search for params dict:
    candidates = [d]
    for key in ("params", "prediction", "pred", "result"):
        if isinstance(d.get(key), dict):
            candidates.append(d[key])
    # First candidate with all keys wins; otherwise partial fill
    best = {}
    for cand in candidates:
        hit = {k: cand.get(k) for k in PARAM_KEYS if k in cand}
        best.update(hit)
    # Fill missing with None
    for k in PARAM_KEYS:
        best.setdefault(k, None)
    return best

def guess_prompt_from_json_or_name(d: dict, json_path: Path, prompts_map: dict):
    # 1) JSON contains prompt?
    for key in ("prompt", "query", "text"):
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        # also check in nested meta
        if isinstance(d.get("meta"), dict):
            inner = d["meta"].get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    # 2) Use filename slug to match a prompt from CSV
    stem = json_path.stem  # without extension
    # Sometimes we produce names like 2025-...__warm_wooden_pluck_short_sustain
    # Try to pick the last chunk after "__"
    if "__" in stem:
        stem_slug = stem.split("__")[-1]
    else:
        stem_slug = stem
    # Try exact match first
    if stem_slug in prompts_map:
        return prompts_map[stem_slug]
    # Try contains
    for slug, prompt in prompts_map.items():
        if slug and slug in stem_slug:
            return prompt
    # Give up: return filename as a placeholder
    return stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="Dir containing text2synth *.json outputs")
    ap.add_argument("--out_csv",  required=True, help="Path to write the consolidated CSV")
    ap.add_argument("--prompts_csv", default=None, help="Optional CSV with a 'query' or 'prompt' column to recover prompts")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    out_csv  = Path(args.out_csv)
    prompts_map = load_prompts_map(Path(args.prompts_csv)) if args.prompts_csv else {}

    rows_out = []
    json_files = sorted(json_dir.rglob("*.json"))
    if not json_files:
        print(f"[ERR] No .json found under {json_dir}")
        return

    for jpath in json_files:
        try:
            d = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Could not read {jpath}: {e}")
            continue

        params = extract_params(d)
        prompt = guess_prompt_from_json_or_name(d, jpath, prompts_map)

        # Find sibling wav
        wav_path = jpath.with_suffix(".wav")
        if not wav_path.exists():
            # also try removing any suffix like ".params.json" -> ".wav"
            if jpath.name.endswith(".params.json"):
                wav_path = jpath.with_name(jpath.name.replace(".params.json", ".wav"))
        wav_str = str(wav_path) if wav_path.exists() else ""

        row = {
            "file_json": str(jpath),
            "file_wav": wav_str,
            "prompt": prompt,
        }
        for k in PARAM_KEYS:
            row[k] = params.get(k)
        rows_out.append(row)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_json","file_wav","prompt"] + PARAM_KEYS
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"[OK] Wrote {len(rows_out)} rows to {out_csv}")

if __name__ == "__main__":
    main()
