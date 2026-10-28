"""Download the public Kaggle mirror of Roboflow Construction Site Safety v28."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "snehilsanyal/construction-site-safety-image-dataset-roboflow"
    "?datasetVersionNumber=3"
)
EXPECTED_SHA256 = "f1acf02c7116a09cdbb2b3ace1ee9b597977c0eb0bd0025e8949f81091fb2a3d"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "Construction-Site-Safety-PPE-Detection-main" / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="Use an already downloaded ZIP")
    args = parser.parse_args()
    temporary = None
    archive_path = args.archive
    if archive_path is None:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as stream:
            archive_path = Path(stream.name)
            temporary = archive_path
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, archive_path)
    try:
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256:
            raise ValueError(f"Unexpected archive SHA-256: {digest}")
        count = 0
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                parts = Path(entry.filename).parts
                if (len(parts) != 4 or parts[0] != "css-data"
                        or parts[1] not in ("train", "valid", "test")
                        or parts[2] not in ("images", "labels") or entry.is_dir()):
                    continue
                target = DEST.joinpath(*parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                count += 1
        if count != 5602:
            raise ValueError(f"Expected 5602 image/label files; found {count}")
        print(f"Extracted {count} files to {DEST}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
