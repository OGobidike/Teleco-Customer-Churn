"""Download, verify and clean the IBM Telco Customer Churn dataset.

The data is never committed. It is downloaded from IBM's own repository (Apache-2.0)
and checked against a pinned SHA-256 fingerprint, so every run trains on exactly the
same 7,043 customers.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
N_ROWS = 7043

TARGET = "Churn"
ID_COLUMN = "customerID"

# Yes/No services that only exist when the customer has the parent service.
# "No internet service" / "No phone service" are kept as their own category:
# a customer without internet is not the same as one who declined online backup.
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def default_data_dir() -> Path:
    return Path(os.environ.get("CHURN_DATA_DIR", "data"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(data_dir: Path | None = None) -> Path:
    """Return the path to the verified CSV, downloading it once if needed."""
    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "Telco-Customer-Churn.csv"
    if not path.exists():
        partial = path.with_suffix(".partial")
        urllib.request.urlretrieve(DATA_URL, partial)  # noqa: S310 (fixed https URL)
        partial.replace(path)
    found = sha256_of(path)
    if found != DATA_SHA256:
        raise ValueError(
            f"{path} has SHA-256 {found}, expected {DATA_SHA256}. "
            "Delete it and run again to re-download."
        )
    return path


def load_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """The CSV exactly as published (every column read as text except the numbers)."""
    return pd.read_csv(fetch(data_dir))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Fix the two real data problems; change no values beyond that.

    1. TotalCharges is text, and 11 rows hold a blank. All 11 have tenure 0:
       brand-new customers who have not been billed yet, so their total is 0
       (not the column mean, which would be about $2,283).
    2. The target becomes 1 (churned) / 0 (stayed).

    SeniorCitizen's 0/1 is relabeled No/Yes so every category column reads the
    same way. Everything else (scaling, encoding) is learned inside the model
    pipeline, from training rows only.
    """
    df = raw.copy()
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = total.isna()
    if not (df.loc[blank, "tenure"] == 0).all():
        raise ValueError("A blank TotalCharges belongs to a customer with tenure > 0.")
    df["TotalCharges"] = total.fillna(0.0)
    if not set(df[TARGET].unique()) <= {"Yes", "No"}:
        raise ValueError(f"Unexpected values in {TARGET}: {sorted(df[TARGET].unique())}")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    for column in df.columns:
        if column not in ("tenure", "MonthlyCharges", "TotalCharges", TARGET):
            df[column] = df[column].astype(str)
    return df


def load(data_dir: Path | None = None) -> pd.DataFrame:
    """The clean dataset, one row per customer, indexed by customer ID."""
    df = clean(load_raw(data_dir))
    if len(df) != N_ROWS or df[ID_COLUMN].duplicated().any():
        raise ValueError("Unexpected rows: the dataset does not match the pinned version.")
    return df.set_index(ID_COLUMN)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]
