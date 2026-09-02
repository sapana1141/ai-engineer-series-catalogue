import argparse
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DB_PATH, EMBED_MODEL


def read_input(file_path):
    """Read the catalogue file."""

    # The supplied file is named .csv but is actually an XLSX file.
    with open(file_path, "rb") as f:
        header = f.read(4)

    if header == b"PK\x03\x04":
        print("Detected Excel-formatted input")
        return pd.read_excel(file_path, engine="openpyxl")

    print("Detected CSV input")
    return pd.read_csv(
        file_path,
        encoding="latin1"
    )


def clean_data(df):
    """Clean catalogue data."""

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Handle missing values
    df["currency"] = df["currency"].fillna("NA")
    df["parent"] = df["parent"].fillna("ROOT")

    # Normalize discontinued flag
    df["discontinued"] = (
        df["discontinued"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "Y": True,
            "N": False
        })
    )

    # Keep title as original human-readable string (stripped)
    df["title"] = df["title"].astype(str).str.strip()

    # Text used for search normalization
    df["search_text"] = (
        df["title"].str.lower().str.replace("-", " ", regex=False) + " " +
        df["category"].fillna("") + " " +
        df["subcategory"].fillna("") + " " +
        df["subset"].fillna("") + " " +
        df["frequency"].fillna("") + " " +
        df["unit"].fillna("") + " " +
        df["currency"].fillna("")
    ).str.strip()

    return df


def ingest(file_path):
    print(f"Reading: {file_path}")

    df = read_input(file_path)

    print(f"Rows read: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    df = clean_data(df)
    
    print("Loading AI Model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Generating vector embeddings (this may take a few seconds)...")
    # 1. Convert the 'search_text' column into a list of vectors
    raw_embeddings = model.encode(df["search_text"].tolist(), show_progress_bar=True)

    # 2. Convert the vectors to binary bytes (BLOB) and add as a DataFrame column
    df["embedding"] = [np.array(emb, dtype=np.float32).tobytes() for emb in raw_embeddings]

    conn = sqlite3.connect(DB_PATH)

    try:
        df.to_sql(
            "series_catalogue",
            conn,
            if_exists="replace",
            index=False
        )

        conn.commit()

    finally:
        conn.close()

    print(f"Loaded {len(df)} records into SQLite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to series_catalogue_raw.csv"
    )

    args = parser.parse_args()

    ingest(args.input)
