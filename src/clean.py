"""
clean.py  --  steam/final_dataset.csv  ->  steam/clean_dataset.csv

Run from the project folder:
    python src/clean.py

What it does:
  - keeps games with at least 100 reviews (decision 13 in CLAUDE.md)
  - release_date -> date; price -> price_usd (dollars; free games = 0)
  - splits genres / categories / developers / publishers into lists ("; " separated)
  - parses SteamSpy tags JSON
It does NOT pick features yet. Columns known only AFTER release are kept for
analysis but listed in LEAKAGE_COLS so they are never used as model inputs.
"""

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "steam", "final_dataset.csv")
OUT = os.path.join(ROOT, "steam", "clean_dataset.csv")

MIN_REVIEWS = 100

# produced after release / by users -> never a model input (decision 1)
LEAKAGE_COLS = ["recommendations_total", "metacritic_score", "dataset_review_count",
                "dataset_review_positive", "steam_pos", "steam_neg", "steam_total",
                "steam_desc", "spy_owners", "n_reviews"]


def split_list(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]


def load_clean():
    df = pd.read_csv(SRC)
    report = [f"final_dataset.csv: {len(df)} games"]

    df["n_reviews"] = df["steam_pos"] + df["steam_neg"]
    df = df[df["n_reviews"] >= MIN_REVIEWS].copy()
    report.append(f"after n_reviews >= {MIN_REVIEWS}: {len(df)} games")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    report.append(f"bad release_date: {df['release_date'].isna().sum()}")

    # Price: the dataset is a 2025 snapshot. initial_price = current full (non-discount)
    # price, the closest thing we have to the launch price. Only USD is comparable.
    usd = df["mat_currency"] == "USD"
    df["price_usd"] = (df["mat_initial_price"] / 100).where(usd)
    df.loc[df["is_free"], "price_usd"] = 0.0
    report.append(f"price_usd missing (non-USD or no price): {df['price_usd'].isna().sum()}")

    df["required_age"] = pd.to_numeric(df["required_age"], errors="coerce")

    for col in ["genres", "categories", "developers", "publishers"]:
        df[col] = df[col].apply(split_list)

    df["tags"] = df["spy_tags_json"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s.strip() else {})
    report.append(f"games without tags: {(df['tags'].apply(len) == 0).sum()}")
    return df, report


if __name__ == "__main__":
    df, report = load_clean()
    out = df.copy()
    for col in ["genres", "categories", "developers", "publishers"]:
        out[col] = out[col].apply("; ".join)
    out["tags"] = out["tags"].apply(lambda d: json.dumps(d, ensure_ascii=False))
    out = out.drop(columns=["spy_tags_json"])
    out.to_csv(OUT, index=False, encoding="utf-8")
    report.append(f"wrote {os.path.relpath(OUT, ROOT)}: {len(out)} rows, {out.shape[1]} columns")
    print("\n".join(report))
