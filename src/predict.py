"""
predict.py  --  predict the Steam positive recommendation ratio of a NEW game (before release)

Step 1 (once): train the final model on ALL games 2006-2025 and save it
    python src/predict.py --train
Step 2: predict from a JSON file describing the game
    python src/predict.py examples/new_game.json
Optional: fewest changes to reach a target rating (price, multiplayer modes, Early Access, and
gameplay/design tags that fit the game's genres):
    python src/predict.py examples/new_game.json --target 90

Final model = XGBoost with the CV-chosen settings (CLAUDE.md decision 19), recency lam = 0.3.
Saved files: models/xgb_final.json (model) + models/meta.json (feature list, fill values,
developer/publisher history, known genres/categories/tags).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean import ROOT, load_clean  # noqa: E402
from features import (DROP_CATEGORIES, OPINION_TAGS, TOP_TAGS, fill_values,  # noqa: E402
                      franchise, history_values, is_same_series, norm_name, series_info, slug,
                      top_tags)

FEATURES = os.path.join(ROOT, "steam", "features.csv")
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "xgb_final.json")
META_FILE = os.path.join(MODEL_DIR, "meta.json")

LAM = 0.3
PARAMS = dict(max_depth=4, learning_rate=0.03, n_estimators=1000, min_child_weight=20,
              subsample=0.8, colsample_bytree=0.5, random_state=42, n_jobs=-1)
ID_COLS = ("appid", "name", "release_date", "rating")
TEST_MAE = 8.30  # XGBoost on the 2024-2025 test games (reports/model_results.txt)
# extra inputs of the SALES model only: past owners (log10) of the developer, publisher, previous entry
SALES_EXTRA = ["dev_prev_mean_log_owners", "dev_prev_max_log_owners", "pub_prev_mean_log_owners",
               "pub_prev_max_log_owners", "prev_entry_log_owners"]
# sales model settings: every tree sees all features (so the sales-history columns are always used);
# out-of-fold MAE 0.383 log10 vs 0.388 with the rating-model settings
SALES_PARAMS = {**PARAMS, "colsample_bytree": 1.0, "max_depth": 6}
# for sequels: final log estimate = 70 % model + 30 % previous entry's owners. On 2,025 past sequels
# this has almost the same error (0.363 vs 0.355 log10) and avoids implausible numbers for big series.
SEQUEL_BLEND = 0.3


def owners_high(s):
    """SteamSpy owners range '20,000 .. 50,000' -> 50000 (upper bound)."""
    if not isinstance(s, str) or ".." not in s:
        return np.nan
    return float(s.split("..")[1].replace(",", "").strip())


def owners_log10(s):
    """'20,000 .. 50,000' -> log10 of the range's geometric middle ('0 .. 20,000' -> 10,000)."""
    if not isinstance(s, str) or ".." not in s:
        return np.nan
    lo, hi = [float(x.replace(",", "").strip()) for x in s.split("..")]
    lo = max(lo, hi / 4)
    return float(np.log10(np.sqrt(lo * hi)))


# ------------------------------------------------------------------------------ train
def train():
    df = pd.read_csv(FEATURES).copy()
    df["price_missing"] = df["price_usd"].isna().astype(int)
    feats = [c for c in df.columns if c not in ID_COLS]
    fills = fill_values(df)
    X = df[feats].fillna(fills)
    w = np.exp(-LAM * (df["year"].max() - df["year"]))
    model = XGBRegressor(**PARAMS)
    model.fit(X, df["rating"], sample_weight=w)

    # rough sales model: log10(SteamSpy owners) from the same pre-release features.
    # Rows where owners are implausibly low for the review count (broken SteamSpy data for many
    # recent games, see reports/sales_effects.txt) are left out.
    clean, _ = load_clean()
    v = clean["spy_owners"].apply(owners_log10)
    v[clean["spy_owners"].apply(owners_high) < 3 * clean["n_reviews"]] = np.nan   # broken SteamSpy rows
    clean["log_owners"] = v
    # sales history (only EARLIER games): developer / publisher past owners, previous series entry
    extra = pd.DataFrame(index=clean.index)
    for col, short in [("developers", "dev"), ("publishers", "pub")]:
        extra[f"{short}_prev_mean_log_owners"], extra[f"{short}_prev_max_log_owners"] = \
            history_values(clean, col, clean["log_owners"])
    _, prev_idx = franchise(clean, return_prev=True)
    extra["prev_entry_log_owners"] = [clean["log_owners"].get(i, np.nan) if pd.notna(i) else np.nan
                                      for i in prev_idx]
    extra["appid"] = clean["appid"]
    s = df[["appid"]].merge(extra.assign(log_owners=clean["log_owners"]), on="appid", how="left")
    sales_feats = feats + SALES_EXTRA
    Xs_all = pd.concat([X.reset_index(drop=True), s[SALES_EXTRA]], axis=1)
    y_sales = s["log_owners"].values
    ok = ~np.isnan(y_sales)
    Xs, ys, ws = Xs_all[ok], y_sales[ok], w[ok]
    oof = np.zeros(len(ys))
    for tr_i, te_i in KFold(5, shuffle=True, random_state=42).split(Xs):
        m = XGBRegressor(**SALES_PARAMS).fit(Xs.iloc[tr_i], ys[tr_i], sample_weight=ws.iloc[tr_i])
        oof[te_i] = m.predict(Xs.iloc[te_i])
    sales_mae = float(np.mean(np.abs(oof - ys)))
    sales_model = XGBRegressor(**SALES_PARAMS).fit(Xs[sales_feats], ys, sample_weight=ws)
    # owners history per developer / publisher (sum, count, max of log10 owners), for new games
    own_hist = {}
    for col in ["developers", "publishers"]:
        h = {}
        for names, val in zip(clean[col], clean["log_owners"]):
            if pd.isna(val):
                continue
            for n in names:
                e = h.setdefault(n, [0.0, 0, -np.inf])
                e[0] += float(val)
                e[1] += 1
                e[2] = max(e[2], float(val))
        own_hist[col] = h
    print(f"Sales model: {int(ok.sum())} games (left out {int((~ok).sum())}), "
          f"out-of-fold MAE {sales_mae:.3f} log10 = typical factor x{10 ** sales_mae:.1f}")

    # history of every developer / publisher over all games (all are before a new game)
    hist = {}
    for col in ["developers", "publishers"]:
        h = {}
        for names, r in zip(clean[col], clean["rating"]):
            for n in names:
                s = h.setdefault(n, [0.0, 0])
                s[0] += float(r)
                s[1] += 1
        hist[col] = h

    # for target suggestions: in which genres is each design tag common (among top-10 tags)?
    tops = clean["tags"].apply(top_tags)
    tag_genres = {}
    for t in DESIGN_TAGS:
        has = tops.apply(lambda l, t=t: t in l)
        tag_genres[t] = [g for g in set(x for l in clean["genres"] for x in l)
                         if (clean["genres"].apply(lambda l, g=g: g in l)).sum() >= 100
                         and has[clean["genres"].apply(lambda l, g=g: g in l)].mean() >= TAG_GENRE_MIN_SHARE]

    # original names of the genres / categories / tags the model knows (for the app's menus)
    def known(col, prefix, source):
        names = {n for l in source for n in l}
        return sorted(n for n in names if prefix + slug(n) in feats)
    choices = {"genres": known("genres", "genre_", clean["genres"]),
               "categories": [c for c in known("categories", "cat_", clean["categories"])
                              if c not in DROP_CATEGORIES],
               "tags": known("tags", "tag_", tops)}

    # all games, to find an earlier entry of the same series (decision 20)
    n_prev = dict(zip(df["appid"], df["series_prev_entries"]))
    games = [[r["name"], r["release_date"].strftime("%Y-%m-%d"), float(r["rating"]),
              int(n_prev.get(r["appid"], 0)), r["developers"], r["publishers"],
              None if pd.isna(r["log_owners"]) else float(r["log_owners"])]
             for _, r in clean.iterrows()]

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_FILE)
    sales_model.save_model(os.path.join(MODEL_DIR, "xgb_sales.json"))
    meta = {"features": feats, "fills": fills, "history": hist, "games": games, "tag_genres": tag_genres, "lam": LAM,
            "sales_factor": round(10 ** sales_mae, 2), "choices": choices, "owners_history": own_hist,
            "default_achievements": float(clean.loc[clean["year"] >= df["year"].max() - 2, "mat_achievement_count"]
                                          .fillna(0).median()),
            "train_years": [int(df["year"].min()), int(df["year"].max())], "n_games": len(df)}
    with open(META_FILE, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    print(f"Trained on {len(df)} games ({meta['train_years'][0]}-{meta['train_years'][1]}), "
          f"{len(feats)} features. Saved {os.path.relpath(MODEL_FILE, ROOT)} and {os.path.relpath(META_FILE, ROOT)}")


# ------------------------------------------------------------------------------ predict
_SERIES = {}


def _series_index(meta):
    """Series info of every known game, computed once: newest-first list and first-word index."""
    if id(meta) not in _SERIES:
        items = [(g, series_info(g[0], g[4], g[5])) for g in meta["games"]]
        by_first = {}
        for g, info in items:
            by_first.setdefault(info["first"], []).append((g, info))
        _SERIES.clear()
        _SERIES[id(meta)] = {"newest_first": sorted(items, key=lambda c: c[0][1], reverse=True),
                             "by_first": by_first}
    return _SERIES[id(meta)]


def game_to_row(game, meta):
    feats, notes = meta["features"], []
    row = dict.fromkeys(feats, 0.0)
    date = pd.to_datetime(game["release_date"])
    row["year"], row["month"] = date.year, date.month
    if date.year > meta["train_years"][1] + 1:
        notes.append(f"release year {date.year} is far after the training data "
                     f"(up to {meta['train_years'][1]}); the year effect is extrapolated")

    is_free = bool(game.get("is_free", False))
    price = 0.0 if is_free else game.get("price_usd")
    row["is_free"] = int(is_free)
    row["price_missing"] = int(price is None)
    row["price_usd"] = meta["fills"]["price_usd"] if price is None else float(price)
    row["required_age"] = float(game.get("required_age", 0))
    # not asked in the app (user: not a meaningful input) -> median of recent games unless given
    ach = game.get("achievements")
    row["achievements"] = float(meta.get("default_achievements", 0) if ach is None else ach)
    row["has_achievements"] = int(row["achievements"] > 0)

    genres = game.get("genres", [])
    row["n_genres"] = len(genres)
    categories = [c for c in game.get("categories", []) if c not in DROP_CATEGORIES]
    tags = [t for t in game.get("tags", []) if t not in OPINION_TAGS][:TOP_TAGS]
    row["has_tags"] = int(bool(tags))
    for prefix, values in [("genre_", genres), ("cat_", categories), ("tag_", tags)]:
        for v in values:
            col = prefix + slug(v)
            if col in row:
                row[col] = 1
            else:
                notes.append(f"unknown {prefix[:-1]} '{v}' (ignored)")

    for key, short in [("developers", "dev"), ("publishers", "pub")]:
        cnt = sum(meta["history"][key].get(n, [0, 0])[1] for n in game.get(key, []))
        tot = sum(meta["history"][key].get(n, [0, 0])[0] for n in game.get(key, []))
        row[f"{short}_prev_games"] = cnt
        row[f"{short}_has_history"] = int(cnt > 0)
        row[f"{short}_prev_mean_rating"] = tot / cnt if cnt else meta["fills"][f"{short}_prev_mean_rating"]
        if cnt:
            notes.append(f"{key}: {cnt} earlier game(s), mean rating {tot / cnt:.1f}")
        else:
            notes.append(f"{key}: no earlier games in the dataset")
    row["self_published"] = int(bool(set(game.get("developers", [])) & set(game.get("publishers", []))))

    # series: "previous_entry" (exact Steam name) if given, otherwise detected from the name
    prev = None
    day = date.strftime("%Y-%m-%d")          # dates are ISO strings, so string order = time order
    idx = _series_index(meta)
    if game.get("previous_entry"):
        want = norm_name(game["previous_entry"])
        prev = next((g for g, info in idx["newest_first"] if g[1] < day and info["nn"] == want), None)
        if prev is None:
            notes.append(f"previous_entry '{game['previous_entry']}' not found in the dataset (ignored)")
    elif "previous_entry" not in game:   # previous_entry "" means: not part of a series
        me = series_info(game.get("name", ""), game.get("developers", []), game.get("publishers", []))
        cands = idx["by_first"].get(me["first"], []) + (idx["by_first"].get(me["key"].split(" ")[0], [])
                                                         if me["key"] else [])
        for g, info in sorted(cands, key=lambda c: c[0][1], reverse=True):
            if g[1] < day and is_same_series(info, me):
                prev = g
                break
    row["prev_entry_rating"] = meta["fills"]["prev_entry_rating"]
    row["years_since_prev_entry"] = meta["fills"]["years_since_prev_entry"]
    if prev:
        row["is_series_entry"] = 1
        row["series_prev_entries"] = prev[3] + 1
        row["years_since_prev_entry"] = (date - pd.to_datetime(prev[1])).days / 365.25
        row["prev_entry_rating"] = prev[2]
        notes.append(f"series: previous entry '{prev[0]}' ({prev[1][:4]}, rating {prev[2]:.1f}), "
                     f"{row['years_since_prev_entry']:.1f} years earlier")
    X = pd.DataFrame([row])[feats]
    X.attrs["prev"] = prev
    return X, notes


def sales_row(X, game, meta):
    """Rating features + past owners of developer / publisher / previous entry (sales model input)."""
    S = X.copy()
    for key, short in [("developers", "dev"), ("publishers", "pub")]:
        h = meta["owners_history"][key]
        es = [h[n] for n in game.get(key, []) if n in h]
        cnt = sum(e[1] for e in es)
        S[f"{short}_prev_mean_log_owners"] = sum(e[0] for e in es) / cnt if cnt else np.nan
        S[f"{short}_prev_max_log_owners"] = max(e[2] for e in es) if es else np.nan
    prev = X.attrs.get("prev")
    S["prev_entry_log_owners"] = prev[6] if prev and len(prev) > 6 and prev[6] is not None else np.nan
    return S[meta["features"] + SALES_EXTRA]


# ------------------------------------------------------------------------------ target rating
PRICES = [4.99, 9.99, 14.99, 19.99, 24.99, 29.99, 39.99, 49.99, 59.99, 69.99]
# multiplayer modes a developer could add (user asked: no Workshop/controller/level editor/achievements)
ADDABLE_CATEGORIES = ["Co-op", "Online Co-op", "Multi-player", "Online PvP", "Shared/Split Screen Co-op",
                      "Shared/Split Screen PvP", "Cross-Platform Multiplayer"]
# gameplay / design changes (Steam tags). Only suggested for a game whose genres already contain
# the tag often enough in past games (meta["tag_genres"]), so suggestions fit the kind of game.
DESIGN_TAGS = ["Co-op", "Online Co-Op", "Local Co-Op", "PvP", "PvE", "Multiplayer", "Story Rich",
               "Choices Matter", "Multiple Endings", "Open World", "Exploration", "Sandbox", "Crafting",
               "Base-Building", "Building", "Survival", "Character Customization", "Procedural Generation",
               "Rogue-lite", "Rogue-like", "Deckbuilding", "Turn-Based Combat", "Turn-Based Strategy",
               "Real Time Tactics", "Tactical", "Stealth", "Physics", "Puzzle", "Metroidvania",
               "Souls-like", "Hack and Slash", "Loot", "Management", "Resource Management", "Economy",
               "Automation", "Colony Sim", "City Builder", "Farming Sim", "Life Sim", "Romance",
               "Dating Sim", "Visual Novel", "Interactive Fiction", "Detective", "Investigation",
               "Psychological Horror", "Survival Horror", "Comedy", "Funny", "Relaxing", "Cozy",
               "Difficult", "Short", "Pixel Graphics", "Hand-drawn", "Cartoony", "Stylized", "Realistic",
               "Retro", "Female Protagonist"]
TAG_GENRE_MIN_SHARE = 0.03  # tag must appear in >= 3 % of past games of one of the game's genres


def candidate_changes(X, feats, meta, genres):
    """All single changes to controllable features: (description, {column: new value})."""
    row = X.iloc[0]
    out = []
    for p in PRICES:
        if not row["is_free"] and abs(p - row["price_usd"]) > 0.5:
            out.append((f"price {row['price_usd']:.2f} -> {p:.2f} USD", {"price_usd": p, "price_missing": 0}))
    for c in ADDABLE_CATEGORIES:
        col = "cat_" + slug(c)
        if col in feats and row[col] == 0:
            out.append((f"add '{c}' mode", {col: 1}))
    if "cat_in_app_purchases" in feats and row["cat_in_app_purchases"] == 1:
        out.append(("remove in-app purchases", {"cat_in_app_purchases": 0}))
    for col in ("genre_early_access", "tag_early_access"):
        if col in feats and row[col] == 1:
            out.append(("release as a finished game instead of Early Access",
                        {c: 0 for c in ("genre_early_access", "tag_early_access") if c in feats}))
            break
    for t in DESIGN_TAGS:
        col = "tag_" + slug(t)
        if col not in feats or t in OPINION_TAGS:
            continue
        if row[col] == 1:
            out.append((f"drop '{t}'", {col: 0}))
        elif set(meta["tag_genres"].get(t, [])) & set(genres):
            out.append((f"add '{t}'", {col: 1, "has_tags": 1}))
    return out


def apply(X, changes):
    Y = X.copy()
    for ch in changes:
        for k, v in ch.items():
            Y[k] = v
    return Y


def suggest(model, X, feats, meta, genres, target, max_steps=6, show=3):
    """Smallest set of changes (by number of changes) that reaches the target rating:
    1) single changes, 2) pairs among the 25 best single changes, 3) greedy, up to max_steps."""
    base = float(model.predict(X)[0])
    cands = candidate_changes(X, feats, meta, genres)
    if not cands:
        return base, []
    singles = pd.concat([apply(X, [ch]) for _, ch in cands], ignore_index=True)
    gain = model.predict(singles[feats]) - base
    order = np.argsort(-gain)
    hits = [i for i in order if base + gain[i] >= target]
    if hits:
        return base, [([cands[i][0]], base + gain[i]) for i in hits[:show]]
    top = [i for i in order[:25] if gain[i] > 0]
    pairs = [(i, j) for a, i in enumerate(top) for j in top[a + 1:]
             if not set(cands[i][1]) & set(cands[j][1])]   # do not change the same column twice
    if pairs:
        P = pd.concat([apply(X, [cands[i][1], cands[j][1]]) for i, j in pairs], ignore_index=True)
        pv = model.predict(P[feats])
        ok = [k for k in np.argsort(-pv) if pv[k] >= target]
        if ok:
            return base, [([cands[pairs[k][0]][0], cands[pairs[k][1]][0]], float(pv[k])) for k in ok[:show]]
    # greedy: add the best remaining change each step
    chosen, cur, Xc = [], base, X
    for _ in range(max_steps):
        cs = [(d, ch) for d, ch in candidate_changes(Xc, feats, meta, genres) if d not in chosen]
        if not cs:
            break
        Y = pd.concat([apply(Xc, [ch]) for _, ch in cs], ignore_index=True)
        v = model.predict(Y[feats])
        k = int(np.argmax(v))
        if v[k] <= cur + 0.05:
            break
        chosen.append(cs[k][0])
        cur, Xc = float(v[k]), apply(Xc, [cs[k][1]])
        if cur >= target:
            break
    return base, [(chosen, cur)] if chosen else []


SALES_NOTE = ("Scenario effects are measured on the RATING only. Their effect on sales could not be "
              "measured with our data (SteamSpy owners are rough and the documented games are mostly "
              "famous ones; see reports/sales_effects.txt).")


def _json(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _xgb(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    m = XGBRegressor()
    m.load_model(path)
    return m


_CACHE = {}


def load_models():
    """Load everything once (the app calls analyze() many times)."""
    if not _CACHE:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError("No trained model yet. Run first:  python src/predict.py --train")
        _CACHE.update(meta=_json("meta.json"), model=_xgb("xgb_final.json"),
                      sales=_xgb("xgb_sales.json"), conf=_json("conformal.json"),
                      qlo=_xgb("xgb_q_lo.json"), qhi=_xgb("xgb_q_hi.json"), scen=_json("scenarios.json"),
                      length=_json("length_effect.json"))
    return _CACHE


def analyze(game, target=None):
    """Everything the program knows about one game, as a dict (used by the CLI and the app)."""
    M = load_models()
    meta, model = M["meta"], M["model"]
    X, notes = game_to_row(game, meta)
    pred = float(np.clip(model.predict(X)[0], 0, 100))
    res = {"name": game.get("name", "(no name)"), "rating": pred, "test_mae": TEST_MAE,
           "notes": notes, "interval": None, "sales": None, "scenarios": [], "scenario_rating": None,
           "length": None, "target": None}

    hours = game.get("main_story_hours")
    if hours and M["length"]:                          # planned length vs genre (src/hltb_effect.py)
        E = M["length"]
        gs = game.get("genres", [])
        main = next((g for g in gs if g not in E["generic_genres"]), gs[0] if gs else None)
        med = E["genre_median_hours"].get(main) or float(np.median(list(E["genre_median_hours"].values())))
        ratio = float(np.clip(float(hours) / med, *E["ratio_clip"]))
        adj = E["slope_per_doubling"] * float(np.log2(ratio))
        res["length"] = {"hours": float(hours), "genre": main, "genre_median": med, "ratio": float(hours) / med,
                         "adjustment": adj, "slope": E["slope_per_doubling"], "slope_ci95": E["slope_ci95"]}

    if M["conf"] and M["qlo"] is not None:            # 90 % interval (CQR, src/conformal.py)
        lo = float(np.clip(M["qlo"].predict(X)[0] - M["conf"]["Q"], 0, 100))
        hi = float(np.clip(M["qhi"].predict(X)[0] + M["conf"]["Q"], 0, 100))
        res["interval"] = {"level": M["conf"]["level"], "low": lo, "high": hi}

    if M["sales"] is not None:                         # rough owners estimate (SteamSpy-based)
        S = sales_row(X, game, meta)
        log_est = float(M["sales"].predict(S)[0])
        if pd.notna(S["prev_entry_log_owners"].iloc[0]):          # sequel: lean 30 % on the previous entry
            log_est = (1 - SEQUEL_BLEND) * log_est + SEQUEL_BLEND * float(S["prev_entry_log_owners"].iloc[0])
        mid = float(10 ** log_est)
        f = meta["sales_factor"]
        ctx = {}   # real past numbers, shown next to the estimate for context
        if pd.notna(S["prev_entry_log_owners"].iloc[0]):
            ctx["previous entry"] = (X.attrs["prev"][0], 10 ** float(S["prev_entry_log_owners"].iloc[0]))
        for short, label in [("dev", "best earlier game of the developer"), ("pub", "best earlier game of the publisher")]:
            v = S[f"{short}_prev_max_log_owners"].iloc[0]
            if pd.notna(v):
                ctx[label] = (None, 10 ** float(v))
        res["sales"] = {"owners": mid, "low": mid / f, "high": mid * f, "factor": f, "context": ctx}

    chosen = [k for k, v in game.get("scenarios", {}).items() if v]
    if chosen and M["scen"]:                           # scenarios (src/scenarios.py)
        total = 0.0
        for k in chosen:
            if k in M["scen"]:
                s = M["scen"][k]
                total += s["effect"]
                res["scenarios"].append({"key": k, **s})
        adj = res["length"]["adjustment"] if res["length"] else 0.0
        res["scenario_rating"] = float(np.clip(pred + adj + total, 0, 100))

    if target is None:
        target = game.get("target_rating")
    if target is not None:                             # fewest changes to reach a target rating
        base, plans = suggest(model, X, meta["features"], meta, game.get("genres", []), float(target))
        res["target"] = {"target": float(target), "base": base,
                         "reached": base >= target or bool(plans and plans[0][1] >= target),
                         "already": base >= target,
                         "plans": [{"changes": s, "rating": float(v)} for s, v in plans]}
    return res


def predict(path, target=None):
    with open(path, encoding="utf-8") as fh:
        game = json.load(fh)
    try:
        r = analyze(game, target)
    except FileNotFoundError as e:
        sys.exit(str(e))
    print(f"Game: {r['name']}")
    print(f"Predicted positive recommendation ratio: {r['rating']:.1f} %")
    print(f"(typical error on 2024-2025 test games: about +/- {TEST_MAE} points)")
    if r["interval"]:
        i = r["interval"]
        print(f"{i['level']:.0%} prediction interval: {i['low']:.1f} % to {i['high']:.1f} %")
    if r["sales"]:
        s = r["sales"]
        print(f"Rough owners estimate (SteamSpy-based): about {s['owners']:,.0f} "
              f"(typical range {s['low']:,.0f} to {s['high']:,.0f})")
        for label, (nm, val) in s["context"].items():
            print(f"  for comparison, {label}{' ' + repr(nm) if nm else ''}: about {val:,.0f} owners")
    if r["length"]:
        g = r["length"]
        print(f"Planned main story {g['hours']:.0f} h = {g['ratio']:.1f}x the {g['genre']} median ({g['genre_median']:.1f} h): "
              f"{g['adjustment']:+.1f} points -> {np.clip(r['rating'] + g['adjustment'], 0, 100):.1f} % "
              f"(+{g['slope']:.2f} per doubling, from HowLongToBeat data)")
    if r["scenarios"]:
        print("Scenarios (association from documented past cases, not a guarantee):")
        for s in r["scenarios"]:
            print(f"  if {s['description']}: about {s['effect']:+.1f} rating points "
                  f"(95% CI {s['ci95'][0]:+.1f} to {s['ci95'][1]:+.1f}, based on {s['n_games']} games)")
        print(f"  -> rough rating with these scenarios: {r['scenario_rating']:.1f} %")
        print("  " + SALES_NOTE)
    for n in r["notes"]:
        print("  note:", n)
    t = r["target"]
    if t:
        print(f"\nTarget rating {t['target']:.1f} % (current prediction {t['base']:.1f} %):")
        if t["already"]:
            print("  the game already reaches the target; no change needed")
        elif not t["plans"]:
            print("  no controllable change raises the prediction")
        else:
            if not t["reached"]:
                print("  target not reachable with up to 6 controllable changes; best found:")
            for p in t["plans"]:
                print(f"  {len(p['changes'])} change(s) -> {p['rating']:.1f} %:  " + " + ".join(p["changes"]))
            print(f"  (model patterns from past games, not a guarantee; typical error about +/- {TEST_MAE} points)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("game_json", nargs="?", help="JSON file describing the new game")
    ap.add_argument("--train", action="store_true", help="train and save the final model")
    ap.add_argument("--target", type=float, default=None, help="target rating: suggest the fewest changes")
    a = ap.parse_args()
    if a.train:
        train()
    elif a.game_json:
        predict(a.game_json, a.target)
    else:
        ap.print_help()
