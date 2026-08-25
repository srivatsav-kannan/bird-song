"""Re-scrape Xeno-canto (API v3) for the four target species.

Protocol:
- Query by scientific name, including taxonomic synonyms.
- 1 request/second politeness delay; full metadata retained for provenance.
- Recordings already present on disk (matched by XC catalog number) are not
  re-downloaded but ARE recorded in the metadata dump so the manifest builder
  can attach quality/license/recordist info to legacy files.
- Restricted recordings (no download URL) are logged and skipped.

Usage:
  XC_API_KEY=... python scripts/scrape_xenocanto.py [--dry-run]

Output:
  data/xc_metadata.json       — full API metadata for every hit
  <AUDIO_ROOT>/xc/<Species>/  — newly downloaded audio, named XC<id>.<ext>
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

API = "https://xeno-canto.org/api/3/recordings"
REPO = Path(__file__).resolve().parents[1]
AUDIO_ROOT = Path.home() / "Datasets" / "Bird Classification" / "Bird Sound" / "Rescrape_XC"

# Scientific names + synonyms still in circulation on recording platforms.
SPECIES_QUERIES = {
    "ForestOwlet": ["Athene blewitti", "Heteroglaux blewitti"],
    "BanasuraLaughingthrush": ["Montecincla jerdoni", "Trochalopteron jerdoni", "Strophocincla jerdoni"],
    "BugunLiocichla": ["Liocichla bugunorum"],
    "JerdonsCourser": ["Rhinoptilus bitorquatus", "Cursorius bitorquatus"],
}


def fetch_all(session, key, query):
    page, out = 1, []
    while True:
        r = session.get(API, params={"query": f'sp:"{query}"', "key": key, "page": page}, timeout=60)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("recordings", []))
        if page >= int(data.get("numPages", 1)):
            return out
        page += 1
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fetch metadata only, no audio downloads")
    args = ap.parse_args()

    key = os.environ.get("XC_API_KEY")
    if not key:
        sys.exit("Set XC_API_KEY (free key from https://xeno-canto.org/account).")

    session = requests.Session()
    session.headers["User-Agent"] = "birdsong-conservation-research/1.0"

    all_meta = {}
    for species, queries in SPECIES_QUERIES.items():
        seen_ids = set()
        recs = []
        for q in queries:
            for rec in fetch_all(session, key, q):
                if rec["id"] not in seen_ids:
                    seen_ids.add(rec["id"])
                    rec["_query"] = q
                    recs.append(rec)
            time.sleep(1.0)
        all_meta[species] = recs
        print(f"{species}: {len(recs)} recordings on Xeno-canto")

    (REPO / "data").mkdir(exist_ok=True)
    with open(REPO / "data" / "xc_metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    if args.dry_run:
        return

    for species, recs in all_meta.items():
        outdir = AUDIO_ROOT / species
        outdir.mkdir(parents=True, exist_ok=True)
        for rec in recs:
            url = rec.get("file")
            if not url:
                print(f"  SKIP (restricted): XC{rec['id']} {species}")
                continue
            ext = re.sub(r"\?.*$", "", rec.get("file-name", "")).rsplit(".", 1)
            ext = ext[1].lower() if len(ext) == 2 else "mp3"
            dest = outdir / f"XC{rec['id']}.{ext}"
            if dest.exists():
                continue
            time.sleep(1.0)
            resp = session.get(url, timeout=180)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                print(f"  downloaded XC{rec['id']} -> {dest.name}")
            else:
                print(f"  FAILED XC{rec['id']}: HTTP {resp.status_code}")


if __name__ == "__main__":
    main()
