"""
features.py  --  build the model table (only pre-release information)

Run from the project folder:
    python src/features.py
Output:
    steam/features.csv   (one row per game: appid, name, release_date, rating, features...)
    reports/features_summary.txt

Decisions used (CLAUDE.md): 1 no leakage, 3 developer/publisher history only from
EARLIER games, 15 keep Early Access, 16 drop Trading Cards, 17 tag rules,
18 drop categories Valve added later (Family Sharing, Remote Play, accessibility, HDR),
20 franchise features from game names (earlier entry of the same series in the dataset).
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean import ROOT, load_clean  # noqa: E402

OUT = os.path.join(ROOT, "steam", "features.csv")
SUMMARY = os.path.join(ROOT, "reports", "features_summary.txt")

MIN_GENRE = 100     # keep genres with at least this many games
MIN_CATEGORY = 50   # same for categories
MIN_TAG = 50        # same for tags (counted among each game's top-10)
TOP_TAGS = 10

OPINION_TAGS = {  # decision 17, group A
    "Great Soundtrack", "Masterpiece", "Classic", "Cult Classic", "Beautiful", "Addictive",
    "Replay Value", "Epic", "Well-Written", "Immersive", "Nostalgia", "Memes", "Illuminati",
    "e-sports", "Experience"}

DROP_CATEGORIES = {
    # decision 16: added after release for successful games
    "Steam Trading Cards",
    # decision 18: added by Valve later / retroactively, so not pre-release information
    "Family Sharing",
    "Remote Play Together", "Remote Play on TV", "Remote Play on Tablet", "Remote Play on Phone",
    # Steam accessibility + HDR labels (introduced 2024; old games have them only if still updated)
    "Custom Volume Controls", "Playable without Timed Input", "Adjustable Difficulty",
    "Save Anytime", "Subtitle Options", "Color Alternatives", "Adjustable Text Size",
    "Camera Comfort", "Stereo Sound", "Surround Sound", "Mouse Only Option",
    "Keyboard Only Option", "Touch Only Option", "Narrated Game Menus",
    "Chat Speech-to-text", "Chat Text-to-speech", "HDR available", "Steam Timeline"}


def fill_values(train):
    """Values for missing features, computed from TRAINING games only (shared by all scripts).
    'No earlier game' columns get the training mean rating; their *_has_history / is_series_entry
    flags tell the model the value is a placeholder."""
    m = float(train["rating"].mean())
    return {"price_usd": float(train["price_usd"].median()),
            "dev_prev_mean_rating": m, "pub_prev_mean_rating": m,
            "prev_entry_rating": m, "years_since_prev_entry": 0.0}


def slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


def multi_hot(series, prefix, min_count):
    counts = series.explode().value_counts()
    keep = counts[counts >= min_count].index
    cols = {f"{prefix}{slug(n)}": series.apply(lambda l, n=n: int(n in l)) for n in keep}
    return pd.DataFrame(cols, index=series.index), list(keep)


def top_tags(d):
    items = [(k, v) for k, v in d.items() if k not in OPINION_TAGS]
    items.sort(key=lambda kv: -kv[1])
    return [k for k, _ in items[:TOP_TAGS]]


def history(df, col):
    """For each game: number of games by the same developer/publisher released BEFORE it,
    and their mean rating. Same-day releases are not counted (strictly earlier)."""
    n_prev = pd.Series(0.0, index=df.index)
    mean_prev = pd.Series(np.nan, index=df.index)
    sums, cnts = {}, {}
    d = df[["release_date", "rating", col]].sort_values("release_date")
    for _, grp in d.groupby("release_date", sort=True):
        for idx, r in grp.iterrows():          # read history first (all same-day games)
            c = sum(cnts.get(n, 0) for n in r[col])
            if c:
                n_prev[idx] = c
                mean_prev[idx] = sum(sums.get(n, 0.0) for n in r[col]) / c
        for _, r in grp.iterrows():            # then add this day's games
            for n in r[col]:
                sums[n] = sums.get(n, 0.0) + r["rating"]
                cnts[n] = cnts.get(n, 0) + 1
    return n_prev, mean_prev


ROMAN = {"ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


def norm_name(s):
    s = re.sub(r"[™®©]", "", str(s)).lower()
    s = re.sub(r"[^a-z0-9: \-']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def series_key(nn):
    """'hades ii' -> ('hades', True); 'warlock 2: the exiled' -> ('warlock', True)."""
    w = re.split(r"\s*[:\-]\s+|\s+-\s*", nn)[0].strip().split()
    marker = len(w) > 1 and ((w[-1].isdigit() and 1 < int(w[-1]) < 30) or w[-1] in ROMAN)
    return " ".join(w[:-1] if marker else w), marker


def is_same_series(earlier, game):
    """earlier / game: dicts with nn, key, marker, studios (developers + publishers)."""
    share = bool(earlier["studios"] & game["studios"])
    same_key = len(earlier["key"]) >= 4 and earlier["key"] == game["key"] and (share or game["marker"])
    prefix = len(earlier["nn"]) >= 4 and game["nn"].startswith(earlier["nn"] + " ") and share
    return same_key or prefix


def series_info(name, developers, publishers):
    nn = norm_name(name)
    key, marker = series_key(nn)
    return {"nn": nn, "key": key, "marker": marker, "studios": set(developers) | set(publishers),
            "first": nn.split(" ")[0] if nn else ""}


def history_values(df, col, values):
    """Like history(), but for any per-game value (e.g. log10 owners): for each game, the mean and
    max of `values` over games by the same developer/publisher released strictly earlier.
    Games whose value is NaN are not counted."""
    mean_prev = pd.Series(np.nan, index=df.index)
    max_prev = pd.Series(np.nan, index=df.index)
    sums, cnts, mx = {}, {}, {}
    d = df[["release_date", col]].assign(v=values).sort_values("release_date")
    for _, grp in d.groupby("release_date", sort=True):
        for idx, r in grp.iterrows():
            c = sum(cnts.get(n, 0) for n in r[col])
            if c:
                mean_prev[idx] = sum(sums.get(n, 0.0) for n in r[col]) / c
                max_prev[idx] = max(mx[n] for n in r[col] if n in mx)
        for _, r in grp.iterrows():
            if pd.isna(r["v"]):
                continue
            for n in r[col]:
                sums[n] = sums.get(n, 0.0) + r["v"]
                cnts[n] = cnts.get(n, 0) + 1
                mx[n] = max(mx.get(n, -np.inf), r["v"])
    return mean_prev, max_prev


def franchise(df, return_prev=False):
    """Decision 20: for each game, the most recent EARLIER game of the same series in the dataset.
    Returns is_series_entry, series_prev_entries, years_since_prev_entry, prev_entry_rating
    (and, with return_prev=True, also the index of that previous entry, NaN if none)."""
    n = len(df)
    out = {"is_series_entry": np.zeros(n), "series_prev_entries": np.zeros(n),
           "years_since_prev_entry": np.full(n, np.nan), "prev_entry_rating": np.full(n, np.nan)}
    prev_idx = np.full(n, np.nan)
    pos = {idx: i for i, idx in enumerate(df.index)}
    by_first = {}  # first word of name -> earlier games (candidates for a match)
    for idx, r in df.sort_values("release_date").iterrows():
        g = series_info(r["name"], r["developers"], r["publishers"])
        g.update(date=r["release_date"], rating=r["rating"], n_prev=0, idx=idx)
        cands = by_first.get(g["first"], []) + (by_first.get(g["key"].split(" ")[0], []) if g["key"] else [])
        for e in sorted(cands, key=lambda e: e["date"], reverse=True):
            if e["date"] < g["date"] and is_same_series(e, g):
                i = pos[idx]
                g["n_prev"] = e["n_prev"] + 1
                out["is_series_entry"][i] = 1
                out["series_prev_entries"][i] = g["n_prev"]
                out["years_since_prev_entry"][i] = (g["date"] - e["date"]).days / 365.25
                out["prev_entry_rating"][i] = e["rating"]
                prev_idx[i] = e["idx"]
                break
        by_first.setdefault(g["first"], []).append(g)
    res = pd.DataFrame(out, index=df.index)
    return (res, pd.Series(prev_idx, index=df.index)) if return_prev else res


def build():
    df, _ = load_clean()
    log = [f"Games: {len(df)}"]

    X = pd.DataFrame(index=df.index)
    X["year"] = df["year"]
    X["month"] = df["release_date"].dt.month
    X["price_usd"] = df["price_usd"]                  # NaN for non-USD (3.8 %)
    X["is_free"] = df["is_free"].astype(int)
    X["required_age"] = df["required_age"].fillna(0)
    # missing achievement count = game has no achievements (only 112 explicit zeros)
    X["achievements"] = df["mat_achievement_count"].fillna(0)
    X["has_achievements"] = (X["achievements"] > 0).astype(int)
    X["n_genres"] = df["genres"].apply(len)

    g, g_names = multi_hot(df["genres"], "genre_", MIN_GENRE)
    cats = df["categories"].apply(lambda l: [c for c in l if c not in DROP_CATEGORIES])
    c, c_names = multi_hot(cats, "cat_", MIN_CATEGORY)
    tags = df["tags"].apply(top_tags)
    t, t_names = multi_hot(tags, "tag_", MIN_TAG)
    X["has_tags"] = (tags.apply(len) > 0).astype(int)

    for col, short in [("developers", "dev"), ("publishers", "pub")]:
        n_prev, mean_prev = history(df, col)
        X[f"{short}_prev_games"] = n_prev
        X[f"{short}_prev_mean_rating"] = mean_prev   # NaN = no earlier game
        X[f"{short}_has_history"] = (n_prev > 0).astype(int)
    X["self_published"] = df.apply(
        lambda r: int(bool(set(r["developers"]) & set(r["publishers"]))), axis=1)
    fr = franchise(df)

    X = pd.concat([X, fr, g, c, t], axis=1)
    out = pd.concat([df[["appid", "name", "release_date", "rating"]], X], axis=1)
    out.to_csv(OUT, index=False, encoding="utf-8")

    log.append(f"Features: {X.shape[1]}  (genres {len(g_names)}, categories {len(c_names)}, tags {len(t_names)})")
    log.append(f"Genres kept: {', '.join(g_names)}")
    log.append(f"Categories kept: {', '.join(c_names)}")
    log.append(f"Developer history: {X['dev_has_history'].mean():.1%} | publisher history: {X['pub_has_history'].mean():.1%}")
    s = X["is_series_entry"] == 1
    log.append(f"Series entries (decision 20): {int(s.sum())} ({s.mean():.1%}); median years since previous entry "
               f"{X.loc[s, 'years_since_prev_entry'].median():.1f}; mean rating series {df.loc[s, 'rating'].mean():.1f} "
               f"vs others {df.loc[~s, 'rating'].mean():.1f}")
    log.append("Missing values:")
    na = X.isna().mean()
    for k, v in na[na > 0].items():
        log.append(f"  {k:<26} {v:.1%}")
    log.append(f"Wrote {os.path.relpath(OUT, ROOT)}: {out.shape[0]} rows, {out.shape[1]} columns")
    text = "\n".join(log)
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    return out


if __name__ == "__main__":
    build()
