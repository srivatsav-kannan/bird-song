"""Build the canonical recording manifest for the four target species.

Sources scanned:
  1. Dataset_Curated/<Species>/{macauley, Xeno Canto/<grade>}  (legacy collection)
  2. Rescrape_XC/<Species>/XC<id>.<ext>                        (fresh scrape, if present)

One row per unique recording. Uniqueness = catalog id (ML numeric id / XC id) with a
sha256 content-hash safety net for physical duplicates. Duration and sample rate come
from ffprobe. XC metadata (license, recordist, date, quality) is joined from
data/xc_metadata.json when the scrape has been run.

Output: data/manifest.csv
"""

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from birdsong.ff import FFPROBE  # noqa: E402
AUDIO_ROOT = Path.home() / "Datasets" / "Bird Classification" / "Bird Sound"
CURATED = AUDIO_ROOT / "Dataset_Curated"
RESCRAPE = AUDIO_ROOT / "Rescrape_XC"

SPECIES_DIRS = {
    "ForestOwlet-Atheneblewitti": "ForestOwlet",
    "BanasuraLaughingthrush-Montecinclajerdoni": "BanasuraLaughingthrush",
    "BugunLiocichla-Liocichlabugunorum": "BugunLiocichla",
    "Jerdon'sCourser-Rhinoptilusbitorquatus": "JerdonsCourser",
}
VALID_EXT = {".wav", ".mp3", ".m4a"}


def probe(path):
    """Return (duration_s, sample_rate) via ffprobe."""
    out = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    info = json.loads(out.stdout or "{}")
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    sr = 0
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            sr = int(s.get("sample_rate", 0) or 0)
            break
    return round(dur, 3), sr


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def catalog_id(path: Path, source: str):
    if source == "ML":
        m = re.match(r"(\d+)", path.stem)
        return f"ML{m.group(1)}" if m else None
    m = re.search(r"XC\s?(\d+)", path.name)
    return f"XC{m.group(1)}" if m else None


def main():
    xc_meta = {}
    meta_path = REPO / "data" / "xc_metadata.json"
    if meta_path.exists():
        raw = json.loads(meta_path.read_text())
        for sp, recs in raw.items():
            for r in recs:
                xc_meta[f"XC{r['id']}"] = r

    rows, seen_ids, seen_hashes = [], set(), {}

    def add(path: Path, species: str, source: str, grade: str):
        cid = catalog_id(path, source)
        h = sha256(path)
        if cid and cid in seen_ids:
            return "dup_id"
        if h in seen_hashes:
            return f"dup_hash_of:{seen_hashes[h]}"
        seen_ids.add(cid)
        seen_hashes[h] = cid or path.name
        dur, sr = probe(path)
        m = xc_meta.get(cid, {})
        rows.append({
            "recording_id": cid or f"HASH{h[:12]}",
            "species": species,
            "source": source,
            "quality_grade": m.get("q", grade),
            "duration_s": dur,
            "sample_rate": sr,
            "license": m.get("lic", ""),
            "recordist": m.get("rec", ""),
            "date": m.get("date", ""),
            "country": m.get("cnt", ""),
            "path": str(path),
            "sha256": h,
        })
        return "ok"

    skipped = []
    for dirname, species in SPECIES_DIRS.items():
        base = CURATED / dirname
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in VALID_EXT or path.name.startswith("."):
                continue
            rel = path.relative_to(base).parts
            if rel[0] == "macauley":
                src, grade = "ML", ""
            elif rel[0] in ("Xeno Canto", "xeno-canto"):
                src, grade = "XC", rel[1] if len(rel) > 2 else ""
            else:
                src, grade = "unknown", ""
            status = add(path, species, src, grade)
            if status != "ok":
                skipped.append((str(path), status))

        rescrape_dir = RESCRAPE / species
        if rescrape_dir.exists():
            for path in sorted(rescrape_dir.iterdir()):
                if path.suffix.lower() not in VALID_EXT:
                    continue
                status = add(path, species, "XC", "")
                if status != "ok":
                    skipped.append((str(path), status))

    out = REPO / "data" / "manifest.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} unique recordings -> {out}")
    from collections import Counter
    for (sp, src), n in sorted(Counter((r["species"], r["source"]) for r in rows).items()):
        print(f"  {sp:24s} {src:3s} {n}")
    if skipped:
        print(f"Skipped {len(skipped)} duplicates:")
        for p, why in skipped:
            print(f"  {why:28s} {p}")


if __name__ == "__main__":
    main()
