"""
app.py  --  Steam Game Rating Predictor (first version of the program, Streamlit web page)

Run from the project folder:
    python -m streamlit run app/app.py
It opens in your browser (http://localhost:8501). Close with Ctrl+C in the window.
Needs the trained models in models/ (python src/predict.py --train, src/conformal.py, src/scenarios.py).
"""

import datetime as dt
import html
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ui  # noqa: E402
from predict import SALES_NOTE, analyze, load_models  # noqa: E402

st.set_page_config(page_title="Steam Rating Predictor", page_icon="🎮", layout="wide")


@st.cache_resource
def models():
    return load_models()


try:
    M = models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
meta, scen = M["meta"], M["scen"] or {}
ch = meta["choices"]


# Streamlit re-sends every menu option on every click. Measured: with the full lists (21,000 studio
# names + 19,600 games) one click took ~0.95 s; with short lists ~0.35 s. So the menus list the most
# common names; any other EXACT Steam name can be typed (accept_new_options) and its history is found.
N_STUDIOS, N_GAMES = 400, 600


@st.cache_data
def studios(key):
    """The developers / publishers with the most games on Steam, most games first."""
    h = meta["history"][key]
    return sorted(h, key=lambda n: (-h[n][1], n.lower()))[:N_STUDIOS]


@st.cache_data
def series_games():
    """The best-selling games (SteamSpy owners), as candidates for 'previous game in the series'."""
    best = sorted((g for g in meta["games"] if g[6] is not None), key=lambda g: -g[6])[:N_GAMES]
    return sorted({g[0] for g in best}, key=str.lower)


@st.cache_data(max_entries=300, show_spinner=False)
def predict_cached(game_json, target):
    """Same inputs -> same answer instantly (e.g. ticking a box off and on again)."""
    return analyze(json.loads(game_json), target)

ui.inject_css()
ui.fix_dropdowns()
ui.hero("Steam Rating Predictor",
        "Predicts the positive recommendation ratio of a PC game on Steam from information known before "
        "release. Khatam University, Machine Learning course project.")

# ------------------------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("🕹️ Your game")
    name = st.text_input("Game name", "My New Game")
    date = st.date_input("Planned release date", dt.date.today() + dt.timedelta(days=180))
    free = st.checkbox("Free to play")
    price = st.number_input("Price (USD)", 0.0, 100.0, 19.99, 1.0, disabled=free)
    # Steam's "required age" is an optional age gate: 96 % of games leave it at 0, even mature ones.
    # The user picks a normal age rating; it is mapped to the closest age-gate value the model knows.
    AGE = {"3+ (everyone)": 0, "7+": 0, "10+": 10, "12+": 12, "13+ (teen)": 13,
           "16+": 16, "17+ (mature)": 17, "18+ (adults only)": 18}
    age = AGE[st.selectbox("Age rating", list(AGE), help="Steam only uses this as an optional age gate; "
                           "most games, even mature ones, have none. 3+ and 7+ mean no age gate.")]
    genres = st.multiselect("Genres", ch["genres"], ["Indie", "Action"])
    categories = st.multiselect("Steam features / modes", ch["categories"], ["Single-player"])
    tags = st.multiselect("Steam tags (most important first, max 10)", ch["tags"], max_selections=10)
    devs = st.multiselect("Developer(s)", studios("developers"), accept_new_options=True,
                          placeholder="Type to search, or type a new studio name",
                          help="The list shows the 400 studios with the most Steam games. Not listed? Type the exact "
                               "Steam name and its track record is still used; an unknown name counts as a new studio.")
    pubs = st.multiselect("Publisher(s)", studios("publishers"), accept_new_options=True,
                          placeholder="Type to search, or type a new publisher name",
                          help="The 400 publishers with the most Steam games. Not listed? Type the exact Steam name.")
    use_len = st.checkbox("I know the planned main-story length")
    hours = st.number_input("Main story length (hours)", 0.5, 500.0, 10.0, 0.5, disabled=not use_len)
    prev = st.selectbox("Previous game in the series", ["(detect from the name)", "(none)"] + series_games(),
                        accept_new_options=True, help="Lists the 600 best-selling games. Not listed? Type the exact "
                                                      "Steam game name.")

    st.header("🎲 What if…")
    st.caption("Effects measured on past games with documented cases (rating only).")
    picked = {}
    for k, s in scen.items():
        picked[k] = st.checkbox(f"{s['description']}  ({s['effect']:+.1f} pts)", key=k)

    st.header("🎯 Target rating")
    use_target = st.checkbox("Suggest the fewest changes to reach a target")
    target = st.slider("Target (%)", 50, 100, 85, disabled=not use_target)

game = {"name": name, "release_date": date.isoformat(), "is_free": free,
        "price_usd": None if free else price, "required_age": age,
        "genres": genres, "categories": categories, "tags": tags,
        "developers": devs, "publishers": pubs, "scenarios": picked,
        "main_story_hours": hours if use_len else None}
if prev == "(none)":
    game["previous_entry"] = ""
elif prev != "(detect from the name)":
    game["previous_entry"] = prev

r = predict_cached(json.dumps(game, sort_keys=True), float(target) if use_target else None)

# ------------------------------------------------------------------------------------ results
ui.gauge(r["rating"], r["interval"], r["test_mae"])
if r["sales"]:
    ui.sales_card(r["sales"])
    st.caption("Sales are hard to predict before release: on past games the estimate is typically off by "
               f"×{r['sales']['factor']}, and big hits are usually under-estimated.")

# result cards: one flow that fills from the left (one card -> left; several -> two columns)
cards = []
if r["length"]:
    g = r["length"]
    new = min(max(r["rating"] + g["adjustment"], 0), 100)
    kind = "pos" if g["adjustment"] >= 0 else "neg"
    cards.append(("length", "⏱️ Planned length",
                  ui.chip(f"{new:.1f} %", "big step") + ui.chip(f"{g['adjustment']:+.1f} pts", kind) +
                  f'<p class="muted">{g["hours"]:.0f} h main story = {g["ratio"]:.1f}× the usual {html.escape(g["genre"])} '
                  f'game ({g["genre_median"]:.1f} h). Longer than usual for the genre goes with slightly higher ratings: '
                  f'about {g["slope"]:+.1f} points per doubling (HowLongToBeat data, 3,637 games).</p>'))
if r["scenarios"]:
    chips = "".join(ui.chip(f"{s['effect']:+.1f}  {s['description'].split(' (')[0]}", "neg" if s["effect"] < 0 else "pos")
                    for s in r["scenarios"])
    detail = "".join(f'<p class="muted">• {html.escape(s["description"])}: 95% CI {s["ci95"][0]:+.1f} to '
                     f'{s["ci95"][1]:+.1f}, {s["n_games"]} past games</p>' for s in r["scenarios"])
    cards.append(("scenarios", "🎲 What-if scenarios",
                  ui.chip(f"{r['scenario_rating']:.1f} %", "big step") +
                  ui.chip(f"{r['scenario_rating'] - r['rating']:+.1f} pts", "neg" if r['scenario_rating'] < r['rating'] else "pos")
                  + "<br>" + chips + detail + f'<p class="muted">ℹ️ {html.escape(SALES_NOTE)}</p>'))
t = r["target"]
if t:
    if t["already"]:
        inner = ui.chip("Already reaches the target 🎉", "pos big")
    elif not t["plans"]:
        inner = '<p class="muted">No controllable change raises the prediction.</p>'
    else:
        inner = "" if t["reached"] else '<p class="muted">⚠️ Not reachable with up to 6 changes; best found:</p>'
        for p in t["plans"]:
            inner += ('<div class="plan">' + ui.chip(f"{len(p['changes'])} change(s) → {p['rating']:.1f} %", "big pos") +
                      "<br>" + "".join(ui.chip(c, "step") for c in p["changes"]) + "</div>")
        inner += ('<p class="muted">Patterns from past games, not a guarantee. Changes considered: price, '
                  'multiplayer / co-op modes, in-app purchases, Early Access, and gameplay tags that fit your genres.</p>')
    cards.append(("target", f"🎯 How to reach {t['target']:.0f} %", inner))
ui.cards("results", cards)

if r["notes"]:
    with st.expander("Details"):
        for n in r["notes"]:
            st.write("-", n)

st.divider()
st.caption("Model: XGBoost trained on 19,612 Steam games (2006–2025) with recency weights. "
           "Test on 2024–2025 games: MAE 8.3, R² 0.26. Ratings are hard to predict before release; "
           "use the result as a rough guide.")
