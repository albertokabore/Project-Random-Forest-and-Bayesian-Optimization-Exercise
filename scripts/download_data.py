"""Download the UCI Heart Disease files (dataset 45) into data/raw/.

Only the four "processed" hospital files and the documentation are kept.
Usage:  python scripts/download_data.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
KEEP = {
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
    "heart-disease.names",
}
EXPECTED_ROWS = {"processed.cleveland.data": 303, "processed.hungarian.data": 294,
                 "processed.switzerland.data": 123, "processed.va.data": 200}
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"
MAX_BYTES = 5_000_000  # the archive is ~130 KB; refuse anything unexpectedly large


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        payload = resp.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        print("Archive larger than expected; aborting.", file=sys.stderr)
        return 1

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in KEEP:
            # Write by basename only, so a crafted archive path cannot escape DEST.
            (DEST / Path(name).name).write_bytes(zf.read(name))

    for name, rows in EXPECTED_ROWS.items():
        n = sum(1 for line in (DEST / name).read_text().splitlines() if line.strip())
        if n != rows:
            print(f"{name}: expected {rows} rows, found {n}", file=sys.stderr)
            return 1
    print(f"Saved {len(KEEP)} files to {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
