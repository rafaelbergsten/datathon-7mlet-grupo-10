"""Download, validate and minimize the Bank Marketing dataset."""

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_PAGE = "https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/henriqueyamahata/bank-marketing"
CSV_NAME = "bank-additional-full.csv"
EXPECTED_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
SAFE_COLUMNS = ["contact", "month", "day_of_week", "previous", "poutcome", "y"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the allowed columns, validate them and map y to a binary target."""
    missing = set(SAFE_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = raw.loc[:, SAFE_COLUMNS].copy()
    if frame.isna().any().any():
        raise ValueError("Missing values require explicit handling before training")
    if not frame["y"].isin(["yes", "no"]).all():
        raise ValueError("Target must contain only yes/no values")
    if not frame["contact"].isin(["cellular", "telephone"]).all():
        raise ValueError("Unsupported historical contact type")
    if (frame["previous"] < 0).any():
        raise ValueError("Previous contact count must be non-negative")
    frame["converted"] = frame.pop("y").map({"yes": 1, "no": 0}).astype("int64")
    return frame


def prepare_data(archive_path: Path | None = None) -> dict:
    """Download the pinned archive, check its hash and save the minimized CSV."""
    if archive_path:
        payload = archive_path.read_bytes()
    else:
        request = Request(DOWNLOAD_URL, headers={"User-Agent": "academic-datathon/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValueError("Kaggle did not return a ZIP. Download the archive manually.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        source = archive.read(CSV_NAME)
    digest = sha256(source)
    if digest != EXPECTED_SHA256:
        raise ValueError("Dataset fingerprint changed; review the new version explicitly")
    raw = pd.read_csv(io.BytesIO(source), sep=";", usecols=SAFE_COLUMNS)
    frame = clean_data(raw)
    path = ROOT / "data/processed/bank_safe.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    metadata = {
        "source": KAGGLE_PAGE,
        "download_url": DOWNLOAD_URL,
        "upstream": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "upstream_doi": "10.24432/C5K306",
        "upstream_license": "CC BY 4.0",
        "source_file": CSV_NAME,
        "kaggle_version": 1,
        "kaggle_last_updated": "2018-06-06T15:19:35.95Z",
        "kaggle_license_label": "Other (specified in description)",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv_sha256": digest,
        "processed_sha256": sha256(path.read_bytes()),
        "rows": len(frame),
        "columns": list(frame.columns),
        "target": "converted: yes=1, no=0",
        "row_order": "Original chronological order preserved",
        "deduplication": "No entity ID: identical minimized rows may be distinct observations",
        "raw_storage": "Archive read in memory; raw data not persisted by this pipeline",
    }
    (ROOT / "data/manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_data() -> pd.DataFrame:
    path = ROOT / "data/processed/bank_safe.csv"
    if not path.exists():
        raise FileNotFoundError("Run python -m banditlab.data first")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if sha256(path.read_bytes()) != manifest["processed_sha256"]:
        raise ValueError("Processed data fingerprint mismatch; regenerate the dataset")
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="Optional manually downloaded Kaggle ZIP")
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.archive), indent=2))
