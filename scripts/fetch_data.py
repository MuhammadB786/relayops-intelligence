"""Download the NASA C-MAPSS data files this project uses into data/.

The project ships no data (data/ is gitignored): the raw text files come
from the official NASA distribution, mirrored by the PHM Society data
repository. That distribution is a zip inside a zip — the outer archive
holds CMAPSSData.zip, which holds the per-subset text files flat. This
script unwraps both layers and extracts exactly the files the pipeline
reads (FD001 for M1–M4, FD002 for the M2 generalization run).

Stdlib only, so it can run in a fresh checkout before anything else —
including as the first step of a deployment build (see render.yaml),
where scripts/train_service.py then trains the service bundle from the
fetched files. Existing files are kept unless --force is given.

Run:  python scripts/fetch_data.py [--data-dir data] [--force]
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

#: Official NASA C-MAPSS distribution, PHM Society mirror (see README).
SOURCE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.%20Turbofan%20Engine%20Degradation%20Simulation%20Data%20Set.zip"
)

#: The files the pipeline reads, by basename inside CMAPSSData.zip.
WANTED = (
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
)


def fetch(data_dir: Path, force: bool = False) -> list[Path]:
    """Download (if needed) and extract the wanted files into data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in WANTED if force or not (data_dir / name).is_file()]
    if not missing:
        print(f"all {len(WANTED)} data files already present in {data_dir}/")
        return [data_dir / name for name in WANTED]

    print(f"downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL) as response:
        outer_bytes = response.read()
    print(f"  {len(outer_bytes) / 1e6:.1f} MB; unwrapping CMAPSSData.zip")
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
        inner_name = next(
            n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        inner_bytes = outer.read(inner_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        available = {Path(n).name for n in inner.namelist()}
        absent = [name for name in missing if name not in available]
        if absent:
            raise SystemExit(f"distribution is missing expected files: {absent}")
        for name in missing:
            (data_dir / name).write_bytes(inner.read(name))
            print(f"  wrote {data_dir / name}")
    return [data_dir / name for name in WANTED]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--force", action="store_true",
                    help="re-download and overwrite files already present")
    args = ap.parse_args()
    fetch(Path(args.data_dir), force=args.force)


if __name__ == "__main__":
    sys.exit(main())
