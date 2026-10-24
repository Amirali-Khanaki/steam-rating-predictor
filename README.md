# 🎮 Steam Rating Predictor

Predicts the **positive recommendation ratio** (share of positive Steam reviews) of a PC game
**before release**, from information known at that time: price, genres, tags, Steam features,
the developer's and publisher's earlier games, the previous game in the series, and the planned length.

Machine Learning course project, Khatam University (Dr. Babak Majidi) — Amirali Khanaki, 2026.

## What the site shows
- **Predicted rating** with a 90 % prediction interval (conformalized quantile regression) and the
  matching Steam label (e.g. "Very Positive").
- **Rough owners estimate** (SteamSpy-based, typically off by about ×2.4).
- **What-if scenarios** measured on documented past games: severe technical problems at launch,
  aggressive microtransactions, "woke"/DEI backlash, other public controversy (effects on the rating only).
- **Target rating**: the fewest changes (price, multiplayer modes, Early Access, gameplay tags that fit
  the genres) that reach a rating you choose.
- Light and dark theme (top-right menu).

## Run it
Windows: install Python (python.org, tick "Add python.exe to PATH"), then double-click `run_app.bat`.

Any system:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```
Then open http://localhost:8501.

## Model
XGBoost regression on 19,612 Steam games (2006–2025), chronological split (test = 2024–2025 games),
recency-weighted training. Test result: MAE 8.3 points, R² 0.26 (baseline MAE 10.2). Ratings are hard
to predict before release, so use the result as a rough guide.

This repository contains only what is needed to run the site (code and trained models in `models/`).
Data sources used for training: Steam Dataset 2025 (Zenodo), Steam user-review API, SteamSpy,
PlayMyData / HowLongToBeat (Zenodo, CC BY 4.0).
