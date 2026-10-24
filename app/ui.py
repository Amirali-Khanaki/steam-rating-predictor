"""
ui.py  --  look and feel of the app: CSS (fonts, colours, soft animations) and animated HTML widgets.

- Light and dark palettes (the user picks the theme in the top-right menu: three dots -> Settings).
- Everything moves with ease-out curves; numbers animate from the previous value, not from zero.
- Cards that disappear (e.g. a scenario is unticked) play the entrance animation in reverse.
- Respects "reduce motion" in the operating system.
- Small fix for Streamlit menus/calendar that re-open when clicked again to close them.
Glass effects are intentionally not used yet (planned as a later step).
"""

import html
import math

import streamlit as st
import streamlit.components.v1 as components

PALETTES = {
    "dark": dict(bg="#0A0D18", card="#141A2E", card2="#1A2140", line="#252D4A", text="#E7EAF3", muted="#9AA3BD",
                 strong="#FFFFFF", track="#0E1326", side1="#0F1426", side2="#0B0F1E", chip="#1B2240", chiph="#232B52",
                 neg="#FCA5A5", pos="#86EFAC", step="#DDD6FE", blob1="rgba(139,92,246,.18)", blob2="rgba(34,211,238,.12)",
                 t1="#FFFFFF", t2="#C4B5FD", t3="#67E8F9", shadow="rgba(0,0,0,.8)", mark="#FFFFFF"),
    "light": dict(bg="#F6F7FB", card="#FFFFFF", card2="#F7F5FF", line="#E3E6F0", text="#1B1F2E", muted="#5B6478",
                  strong="#11131C", track="#ECEEF6", side1="#F0F1F8", side2="#E9EBF5", chip="#F1EEFF", chiph="#E6E0FF",
                  neg="#B91C1C", pos="#15803D", step="#6D28D9", blob1="rgba(139,92,246,.12)", blob2="rgba(34,211,238,.10)",
                  t1="#4C1D95", t2="#7C3AED", t3="#0891B2", shadow="rgba(30,30,60,.18)", mark="#1B1F2E"),
}


def theme_type():
    try:
        t = st.context.theme.type
    except Exception:
        t = None
    return t if t in PALETTES else "dark"


# Colours live in CSS variables for BOTH themes. A small script marks the page <html data-theme="light|dark">
# from Streamlit's current text colour, so switching theme in the menu recolours everything at once
# (Streamlit does not rerun the script on a theme switch).
VARS = {k: f"var(--{k})" for k in PALETTES["dark"]}


def _var_blocks(sel_dark=":root", sel_light='html[data-theme="light"]'):
    blk = lambda p: ";".join(f"--{k}:{v}" for k, v in p.items())
    return f"{sel_dark}{{{blk(PALETTES['dark'])}}} {sel_light}{{{blk(PALETTES['light'])}}}"


THEME_JS = """
 function __themeOf(doc){ const el = doc.querySelector('[data-testid="stApp"]') || doc.body;
   const m = getComputedStyle(el).color.match(/\\d+/g); if(!m) return 'dark';
   const lum = (0.299*m[0] + 0.587*m[1] + 0.114*m[2]) / 255; return lum < 0.5 ? 'light' : 'dark'; }
"""


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

:root { --accent:#8B5CF6; --accent2:#22D3EE; --ease:cubic-bezier(.22,1,.36,1); }

html, body, [class*="css"], .stMarkdown, p, label, input, textarea { font-family:'Inter',sans-serif; }
h1, h2, h3, .hero-title { font-family:'Space Grotesk','Inter',sans-serif !important; letter-spacing:-.01em; }

.stApp { background:
   radial-gradient(900px 500px at 85%% -10%%, %(blob1)s, transparent 60%%),
   radial-gradient(700px 450px at -10%% 110%%, %(blob2)s, transparent 60%%), %(bg)s; }

/* Streamlit's "running" icons (cyclist, runner...) in the top-right corner: hidden */
[data-testid="stStatusWidget"] { display:none !important; }

/* while Streamlit recomputes it dims the old page; that made every click feel slow and jumpy */
[data-stale="true"], .stale-element { opacity:1 !important; filter:none !important; transition:none !important; }

/* start higher, and line up the sidebar's first title with the page title */
.block-container { padding-top:2.4rem !important; }
[data-testid="stSidebarUserContent"] { padding-top:5px !important; }   /* measured: lines up with the page title */
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:first-child h2 { margin-top:0 !important; }

/* title: icon is outside the gradient text so it keeps its own colours */
.hero { display:flex; align-items:center; gap:.6rem; margin:0 0 .3rem; }
.hero-icon { font-size:2.3rem; line-height:1; animation:bob 3.2s ease-in-out infinite; display:inline-block; }
.hero-title { font-size:2.6rem; font-weight:700; line-height:1.1;
   background:linear-gradient(90deg,%(t1)s 0%%,%(t2)s 30%%,%(t3)s 50%%,%(t2)s 70%%,%(t1)s 100%%); background-size:300%% 100%%;
   -webkit-background-clip:text; background-clip:text; color:transparent;
   animation:shine 6s ease-in-out infinite alternate; }        /* back and forth: no jump at the loop */
.hero-sub { color:var(--muted); margin-bottom:1.2rem; }   /* no entrance: Streamlit redraws it on every click */

/* result cards: one flow from the left; one smooth motion (height + fade + slide together) */
.cards { column-count:2; column-gap:1.4rem; }
@media (max-width:900px) { .cards { column-count:1; } }
.cardwrap { display:grid; grid-template-rows:1fr; break-inside:avoid; margin-bottom:.9rem; }
.cardwrap > .clip { min-height:0; overflow:hidden; padding:4px 2px 10px; margin:-4px -2px -10px; }
.cardwrap.enter { animation:open .5s var(--ease) both; }
.cardwrap.leaving { animation:close .45s var(--ease) both; pointer-events:none; }
.card { background:linear-gradient(180deg,var(--card2),var(--card)); border:1px solid var(--line); border-radius:18px;
   padding:1rem 1.2rem; box-shadow:0 10px 30px -18px %(shadow)s;
   transition:transform .35s var(--ease), box-shadow .35s var(--ease), border-color .35s var(--ease), background .4s, color .4s; }
.card:hover { transform:translateY(-3px); border-color:rgba(139,92,246,.55); box-shadow:0 16px 40px -18px rgba(139,92,246,.55); }
.card .plan { margin-bottom:.9rem; }
.card h4 { margin:0 0 .6rem; font-family:'Space Grotesk',sans-serif; font-size:1.05rem; color:var(--strong); }
.muted { color:var(--muted); font-size:.88rem; }

/* chips */
.chip { display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .7rem; margin:.18rem .25rem .18rem 0;
   border-radius:999px; font-size:.85rem; font-weight:600; border:1px solid var(--line); background:var(--chip); color:var(--strong);
   transition:transform .25s var(--ease), background .25s; }   /* no own animation: moves with its card */
.chip:hover { transform:translateY(-2px) scale(1.03); background:var(--chiph); }
.chip.neg { color:%(neg)s; border-color:rgba(248,113,113,.45); }
.chip.pos { color:%(pos)s; border-color:rgba(74,222,128,.45); }
.chip.step { color:%(step)s; border-color:rgba(139,92,246,.45); }
.chip.big { font-size:1.05rem; padding:.35rem .9rem; }

/* native widgets */
[data-testid="stExpander"] { border-radius:14px !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [role="group"] { transition:border-color .25s var(--ease), box-shadow .25s var(--ease); }
.stCheckbox label { transition:color .2s; }
[data-testid="stSlider"] [role="slider"] { transition:transform .2s var(--ease); }
[data-testid="stSlider"] [role="slider"]:hover { transform:scale(1.25); }

/* sidebar: more room between section titles and what follows */
[data-testid="stSidebar"] { background:linear-gradient(180deg,%(side1)s,%(side2)s) !important; border-right:1px solid var(--line); }
[data-testid="stSidebar"] h2 { font-size:1.05rem !important; padding:.15rem 0 .15rem .65rem !important; border-left:3px solid var(--accent);
   margin:1.4rem 0 .9rem !important; }
[data-testid="stSidebar"] [data-testid="stHeading"] { margin-bottom:.35rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { margin-bottom:.5rem; }

::-webkit-scrollbar { width:10px; } ::-webkit-scrollbar-thumb { background:rgba(120,130,170,.35); border-radius:10px; }

@keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:none; } }
@keyframes open  { from { grid-template-rows:0fr; opacity:0; transform:translateY(10px); margin-bottom:0; }
                   to   { grid-template-rows:1fr; opacity:1; transform:none; margin-bottom:.9rem; } }
@keyframes close { from { grid-template-rows:1fr; opacity:1; transform:none; margin-bottom:.9rem; }
                   to   { grid-template-rows:0fr; opacity:0; transform:translateY(10px); margin-bottom:0; } }
@keyframes pop { from { opacity:0; transform:scale(.85); } to { opacity:1; transform:none; } }
@keyframes shine { from { background-position:0%% 50%%; } to { background-position:100%% 50%%; } }
@keyframes bob { 0%%,100%% { transform:translateY(0) rotate(0); } 50%% { transform:translateY(-5px) rotate(-6deg); } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
</style>
"""


def inject_css():
    css = CSS % VARS
    css = css.replace("<style>", "<style>\n" + _var_blocks() + "\n", 1)
    st.markdown(css, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><span class="hero-icon">🎮</span><span class="hero-title">{html.escape(title)}</span></div>'
                f'<div class="hero-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def steam_label(r):
    """Steam's review-summary words for a positive ratio (rounded thresholds); colours readable in both themes."""
    if r >= 95: return "Overwhelmingly Positive", "#3B9EE0"
    if r >= 70: return ("Very Positive" if r >= 80 else "Mostly Positive"), "#3B9EE0"
    if r >= 40: return "Mixed", "#A88A45"
    if r >= 20: return "Mostly Negative", "#D9542B"
    return "Overwhelmingly Negative", "#C93B33"


def _base(p=VARS):
    return f"""
<style>
 {_var_blocks()}
 html{{transition:color .4s}} .wrap,.bar,.big,.lbl{{transition:background .4s,border-color .4s,color .4s}}
 html,body{{margin:0;background:transparent;font-family:'Inter',system-ui,sans-serif;color:{p['text']};overflow:hidden}}
 .wrap{{display:flex;gap:28px;align-items:center;padding:18px 22px;border-radius:22px;border:1px solid {p['line']};
   background:linear-gradient(135deg,{p['card2']},{p['card']});position:relative;overflow:hidden;
   box-shadow:0 10px 30px -18px {p['shadow']};animation:in .8s cubic-bezier(.22,1,.36,1) both}}
 .dot{{position:absolute;border-radius:50%;background:radial-gradient(circle,#8B5CF6 0,transparent 70%);opacity:.3;
   animation:float 9s ease-in-out infinite}}
 .lbl{{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:{p['muted']}}}
 .big{{font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:700;color:{p['strong']}}}
 .pill{{display:inline-block;margin-top:6px;padding:5px 12px;border-radius:999px;font-weight:700;font-size:.9rem;
   transition:color .6s,border-color .6s}}
 .bar{{position:relative;height:12px;border-radius:999px;background:{p['track']};border:1px solid {p['line']};margin:10px 0 4px}}
 .fill{{position:absolute;top:0;bottom:0;border-radius:999px;background:linear-gradient(90deg,#8B5CF6,#22D3EE);
   transition:left 1.1s cubic-bezier(.22,1,.36,1),width 1.1s cubic-bezier(.22,1,.36,1)}}
 .mark{{position:absolute;top:-4px;width:4px;height:20px;border-radius:2px;background:{p['mark']};box-shadow:0 0 10px {p['mark']};
   transition:left 1.1s cubic-bezier(.22,1,.36,1)}}
 .ticks{{display:flex;justify-content:space-between;font-size:.75rem;color:{p['muted']}}}
 @keyframes in{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}}
 @keyframes float{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate(18px,-14px)}}}}
 @keyframes spark{{0%{{opacity:0;transform:scale(.2) rotate(0)}}40%{{opacity:1}}100%{{opacity:0;transform:scale(1.4) rotate(90deg)}}}}
 @media (prefers-reduced-motion: reduce){{*{{animation:none!important;transition:none!important}}}}
 /* phones / narrow screens: ring + label on the first row, the interval gets its own full-width row */
 @media (max-width:640px){{
   .wrap{{flex-wrap:wrap;gap:14px 18px;padding:16px}}
   .ring{{width:128px!important;height:128px!important}} .ring .big{{font-size:1.55rem}}
   .iv{{flex:1 1 100%!important;min-width:0!important}} .hide-narrow{{display:none}} }}
</style>
<script>
 // the iframe gets exactly the height of its card (no empty gap, nothing cut off), also after resizing
 function __fit(){{ try {{ const f = window.frameElement, w = document.querySelector('.wrap'); if (!f || !w) return;
   const h = Math.ceil(w.getBoundingClientRect().height) + 6; if (Math.abs((parseInt(f.style.height)||0) - h) > 1) {{
   f.style.height = h + 'px'; f.setAttribute('height', h);
   // Streamlit's wrappers around the iframe keep the old fixed height: resize them too
   let el = f.parentElement;
   for (let i = 0; el && i < 3; i++, el = el.parentElement) {{
     el.style.height = h + 'px'; if (el.getAttribute('data-testid') === 'stElementContainer') break; }} }} }} catch(e) {{}} }}
 addEventListener('load', __fit); addEventListener('resize', __fit);
 document.addEventListener('DOMContentLoaded', () => {{ __fit(); const w = document.querySelector('.wrap');
   if (w && window.ResizeObserver) new ResizeObserver(__fit).observe(w); }});
</script>
<script>
 // follow the page theme (set on the parent page by fix_dropdowns / theme watcher)
 function __syncTheme(){{ try {{ const t = window.parent.document.documentElement.dataset.theme || 'dark';
   if (document.documentElement.dataset.theme !== t) document.documentElement.dataset.theme = t; }} catch(e) {{}} }}
 __syncTheme(); setInterval(__syncTheme, 300);
 const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
 function countUp(el, from, to, dec, ms){{
   const f = v => v.toLocaleString(undefined,{{minimumFractionDigits:dec,maximumFractionDigits:dec}});
   if(reduce){{ el.textContent = f(to); return; }}
   const t0 = performance.now();
   function step(t){{ const q = Math.min(1,(t-t0)/ms), e = 1-Math.pow(1-q,4);
     el.textContent = f(from + (to-from)*e); if(q<1) requestAnimationFrame(step); }}
   requestAnimationFrame(step); }}
 // start every moving part at its previous value, then move it to the new one
 function settle(fn){{ requestAnimationFrame(()=>requestAnimationFrame(fn)); }}
</script>"""


def _prev(key, value):
    """Previous value shown for this widget (so animations continue from there), then remember the new one."""
    old = st.session_state.get("_prev_" + key)
    st.session_state["_prev_" + key] = value
    return value if old is None else old


def gauge(rating, interval=None, test_mae=8.3):
    """Animated ring + number from the previous prediction, Steam-style label and the 90 % interval bar."""
    p = VARS
    lab, col = steam_label(rating)
    first = st.session_state.get("_prev_rating") is None
    old = 0.0 if first else _prev("rating", rating)
    if first:
        _prev("rating", rating)
    lo, hi = (interval["low"], interval["high"]) if interval else (rating, rating)
    olo, ohi = _prev("ivlo", lo), _prev("ivhi", hi)
    if first:
        olo, ohi = lo, lo
    r = 62
    circ = 2 * math.pi * r
    off = lambda v: circ * (1 - v / 100)
    sparkle = ""
    if rating >= 85:
        sparkle = "".join(
            f'<div style="position:absolute;left:{x}px;top:{y}px;font-size:{s}px;color:#FBBF24;animation:spark 1.6s {d}s ease-out both">✦</div>'
            for x, y, s, d in [(40, 18, 14, 1.0), (150, 10, 10, 1.2), (165, 120, 12, 1.4), (20, 128, 9, 1.6)])
    iv = ""
    if interval:
        iv = f"""
      <div class="iv" style="flex:1;min-width:220px">
        <div class="lbl">{interval['level']:.0%} prediction interval</div>
        <div class="big" style="font-size:1.7rem"><span id="lo">{olo:.0f}</span>–<span id="hi">{ohi:.0f}</span><span style="font-size:1rem;opacity:.7"> /100</span></div>
        <div class="bar"><div id="fill" class="fill" style="left:{olo}%;width:{max(ohi - olo, 0)}%"></div>
             <div id="mark" class="mark" style="left:calc({old}% - 2px)"></div></div>
        <div class="ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
        <div class="lbl hide-narrow" style="margin-top:10px;text-transform:none;letter-spacing:0">
          Typical error ±{test_mae} points · bright mark = prediction</div>
      </div>
      <script>settle(()=>{{ const fl=document.getElementById('fill'); fl.style.left='{lo}%'; fl.style.width='{hi - lo}%';
        document.getElementById('mark').style.left='calc({rating}% - 2px)';
        countUp(document.getElementById('lo'),{olo},{lo},0,1100); countUp(document.getElementById('hi'),{ohi},{hi},0,1100); }});</script>"""
    body = f"""{_base(p)}
    <div class="wrap" style="{'' if first else 'animation:none'}">
      <div class="dot" style="width:220px;height:220px;right:-60px;top:-80px"></div>
      <div class="dot" style="width:160px;height:160px;left:35%;bottom:-90px;background:radial-gradient(circle,#22D3EE 0,transparent 70%);animation-delay:-4s"></div>
      <div class="ring" style="position:relative;width:170px;height:170px;flex:none">
        <svg width="100%" height="100%" viewBox="0 0 170 170">
          <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#8B5CF6"/><stop offset="1" stop-color="#22D3EE"/></linearGradient>
            <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
          <circle cx="85" cy="85" r="{r}" fill="none" style="stroke:var(--track)" stroke-width="14"/>
          <circle id="arc" cx="85" cy="85" r="{r}" fill="none" stroke="url(#g)" stroke-width="14" stroke-linecap="round"
            transform="rotate(-90 85 85)" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off(old):.1f}" filter="url(#glow)"
            style="transition:stroke-dashoffset 1.4s cubic-bezier(.22,1,.36,1)"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
          <div class="big"><span id="num">{old:.1f}</span></div>
          <div class="lbl">out of 100</div>
        </div>
        {sparkle}
      </div>
      <div style="flex:none">
        <div class="lbl">Steam would show</div>
        <div class="pill" style="color:{col};border:1px solid {col}66;background:{col}1A">{lab}</div>
        <div class="lbl hide-narrow" style="margin-top:14px;text-transform:none;letter-spacing:0">score = share of positive Steam reviews</div>
      </div>
      {iv}
    </div>
    <script>
      settle(()=>{{ document.getElementById('arc').style.strokeDashoffset='{off(rating):.1f}'; }});
      countUp(document.getElementById('num'), {old:.4f}, {rating:.4f}, 1, 1400);
    </script>"""
    components.html(body, height=215)


def sales_card(s):
    """Owners estimate on a log scale (1 thousand .. 100 million), animated from the previous estimate."""
    p = VARS
    pos = lambda v: max(0.0, min(100.0, (math.log10(max(v, 1)) - 3) / 5 * 100))
    first = st.session_state.get("_prev_own") is None
    o_own, o_lo, o_hi = _prev("own", s["owners"]), _prev("ownlo", s["low"]), _prev("ownhi", s["high"])
    if first:
        o_own = 0
    marks = "".join(
        f'<div title="{html.escape(lbl)}" style="position:absolute;top:-6px;left:calc({pos(v)}% - 6px);width:12px;height:24px;'
        f'border-radius:4px;background:#F59E0B;opacity:.85"></div>'
        for lbl, (nm, v) in s["context"].items())
    legend = " · ".join(f"{html.escape(lbl)}{' (' + html.escape(nm) + ')' if nm else ''}: ~{v:,.0f}"
                        for lbl, (nm, v) in s["context"].items())
    body = f"""{_base(p)}
    <div class="wrap" style="display:block;{'' if first else 'animation:none'}">
      <div class="lbl">Rough owners estimate (SteamSpy-based)</div>
      <div class="big">~<span id="own">{o_own:,.0f}</span></div>
      <div class="bar"><div id="fill" class="fill" style="left:{pos(o_lo)}%;width:{pos(o_hi) - pos(o_lo)}%"></div>
        <div id="mark" class="mark" style="left:calc({pos(o_own if not first else s['owners'])}% - 2px)"></div>{marks}</div>
      <div class="ticks"><span>1K</span><span>10K</span><span>100K</span><span>1M</span><span>10M</span><span>100M</span></div>
      <div class="lbl" style="margin-top:8px;text-transform:none;letter-spacing:0">
        Range {s['low']:,.0f} – {s['high']:,.0f} (typical error ×{s['factor']}){' · orange = ' + legend if legend else ''}</div>
    </div>
    <script>
      settle(()=>{{ const f=document.getElementById('fill'); f.style.left='{pos(s['low'])}%'; f.style.width='{pos(s['high']) - pos(s['low'])}%';
        document.getElementById('mark').style.left='calc({pos(s['owners'])}% - 2px)'; }});
      countUp(document.getElementById('own'), {o_own:.0f}, {s['owners']:.0f}, 0, 1400);
    </script>"""
    components.html(body, height=190)


def cards(group, items):
    """All result cards in ONE flow that fills from the left (one card sits on the left; several make
    two columns). A card that was shown last time but is gone now plays the entrance in reverse once.
    Cards that stay keep still (no replay); new cards open smoothly. items: list of (key, title, html)."""
    shown = st.session_state.setdefault("_cards_" + group, {})
    order = st.session_state.setdefault("_order_" + group, [])
    for k, _, _ in items:
        if k not in order:
            order.append(k)
    current = {k: (t, h) for k, t, h in items}
    parts = []
    for k in order:
        if k in current:
            t, h = current[k]
            cls = "stay" if k in shown else "enter"
            parts.append(f'<div class="cardwrap {cls}"><div class="clip"><div class="card"><h4>{t}</h4>{h}</div></div></div>')
            shown[k] = (t, h)
        elif k in shown:                     # just removed: animate it out, then forget it
            t, h = shown.pop(k)
            parts.append(f'<div class="cardwrap leaving"><div class="clip"><div class="card"><h4>{t}</h4>{h}</div></div></div>')
    if parts:
        st.markdown('<div class="cards">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def chip(text, kind="", delay=0.0):
    return f'<span class="chip {kind}" style="animation-delay:{delay}s">{html.escape(text)}</span>'


def fix_dropdowns():
    """Runs once in the page: (1) Streamlit menus (selectbox / multiselect) and the date calendar re-open
    when you click them again to close them -- this closes them instead; (2) watches the light/dark theme."""
    components.html("""
<script>""" + THEME_JS + """
(function(){
  const w = window.parent, doc = w.document;
  if (w.__menuFix) return; w.__menuFix = true;
  // theme watcher: mark <html data-theme="light|dark"> so all CSS variables switch together
  function mark(){ const t = __themeOf(doc); if (doc.documentElement.dataset.theme !== t) doc.documentElement.dataset.theme = t; }
  mark(); setInterval(mark, 250);
  let swallowUntil = 0;
  function openMenu(t){
    // month / year menus inside the date calendar
    const inCal = t.closest('[data-testid="stDateInputCalendar"]');
    if (inCal){ const trg = t.closest('[aria-expanded="true"]');
      return trg ? {kind:'popup', input: trg} : null; }
    const box = t.closest('[data-testid="stSelectbox"],[data-testid="stMultiSelect"]');
    if (box){ const o = box.querySelector('[data-open="true"]');
      if (o && !t.closest('button[aria-label^="Remove"]') && !t.closest('[role="listbox"],[role="option"]'))
        return {kind:'menu', input: box.querySelector('input')}; }
    const d = t.closest('[data-testid="stDateInput"]');
    if (d){ const b = d.querySelector('[data-testid="stDateInputCalendarButton"]');
      if (b && b.getAttribute('aria-expanded') === 'true') return {kind:'date', input: b}; }
    return null; }
  function close(m){
    const esc = new KeyboardEvent('keydown', {key:'Escape', code:'Escape', keyCode:27, bubbles:true, cancelable:true});
    if (m.kind === 'popup'){            // close only the small month/year list, keep the calendar open
      const lb = [...doc.querySelectorAll('[role="listbox"]')].pop();
      (lb || doc.activeElement || m.input).dispatchEvent(esc); return; }
    (m.kind === 'date' ? (doc.querySelector('[data-testid="stDateInputCalendar"]') || m.input) : m.input).dispatchEvent(esc);
    if (m.input && m.input.blur) m.input.blur(); }
  function down(e){ const m = openMenu(e.target); if (!m) return;
    e.preventDefault(); e.stopImmediatePropagation(); swallowUntil = Date.now() + 700; close(m); }
  function swallow(e){ if (Date.now() < swallowUntil){ e.preventDefault(); e.stopImmediatePropagation(); } }
  doc.addEventListener('pointerdown', down, true);
  doc.addEventListener('mousedown', swallow, true);
  ['pointerup','mouseup','click'].forEach(ev => doc.addEventListener(ev, swallow, true));
})();
</script>""", height=0)
