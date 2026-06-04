"""
2026 WC HTML visualizer:
  Section 1 — Group stage tables (12 groups, 4-column grid)
  Section 2 — Knockout bracket with SVG connector lines
"""

import pandas as pd
from pathlib import Path

df = pd.read_csv("data/predictions.csv")
GS = list("ABCDEFGHIJKL")

# ── helpers ──────────────────────────────────────────────────────────────

def ts(t):
    return df[df["team"] == t].iloc[0]

def best(group, col):
    sub = df[df["group"] == group]
    return sub.loc[sub[col].idxmax(), "team"]

def win_color(pw, alpha=1.0):
    t = min(pw / 12.0, 1.0)
    r, g, b = int(230 - 80*t), int(248 - 30*t), int(225 - 90*t)
    return f"rgba({r},{g},{b},{alpha})"

# ── bracket construction ─────────────────────────────────────────────────

F1 = {g: best(g, "p_1st")  for g in GS}
F2 = {g: best(g, "p_2nd") for g in GS}
F3 = {g: best(g, "p_3rd") for g in GS}

# Best 8 thirds by p_qualified of their group's likely 3rd-placer
best8_thirds = [F3[g] for g in sorted(GS, key=lambda g: ts(F3[g])["p_qualified"], reverse=True)[:8]]

# R32 matchups (16 total, 8 per side)
R32_L = [(F1[GS[i]], best8_thirds[i]) for i in range(8)]
R32_R = ([(F1[GS[8+i]], F2[GS[i]]) for i in range(4)] +
         [(F2[GS[4+i]], F2[GS[8+i]]) for i in range(4)])

def adv(a, b):
    return a if ts(a)["p_winner"] >= ts(b)["p_winner"] else b

def nxt(ms):
    w = [adv(a, b) for a, b in ms]
    return [(w[i], w[i+1]) for i in range(0, len(w), 2)]

R16_L  = nxt(R32_L);   R16_R  = nxt(R32_R)
QF_L   = nxt(R16_L);   QF_R   = nxt(R16_R)
SF_L   = nxt(QF_L);    SF_R   = nxt(QF_R)
FIN_L  = adv(*SF_L[0]); FIN_R  = adv(*SF_R[0])
CHAMP  = adv(FIN_L, FIN_R)

# For rendering: left side ordered R32→SF, right side ordered R32→SF
L_ROUNDS = [R32_L, R16_L, QF_L, SF_L]
R_ROUNDS = [R32_R, R16_R, QF_R, SF_R]

STAGE_COLS = ["p_qualified", "p_r16", "p_qf", "p_sf"]

# ── bracket geometry ──────────────────────────────────────────────────────

BH      = 768   # total bracket height
SLOT    = 34    # one team slot height px
GAP     = 2     # gap between slots in a match
MATCH_H = SLOT * 2 + GAP   # 70 px
CW      = 148   # card width
CON     = 22    # connector width (space between card edge and next card)
RW      = CW + CON          # round column width = 170 px
FW      = 190   # final column width

# Total bracket width = 4 rounds left + final + 4 rounds right
TOTAL_W = 4 * RW + FW + 4 * RW   # 1550 px

def tops(n):
    """Top-y for n matches evenly distributed across BH."""
    s = BH / n
    return [s * i + (s - MATCH_H) / 2 for i in range(n)]

# Precompute all match top positions per round size
TOPS = {n: tops(n) for n in [1, 2, 4, 8]}

# For global vertical consistency, R32 uses tops(16): 16 matches across the full height
# Left matches use slots 0-7, right matches use slots 8-15
TOPS16 = tops(16)

def global_tops(round_idx, side):
    """
    Return top-y positions for matches at round_idx (0=R32, 1=R16, 2=QF, 3=SF),
    using globally consistent vertical spacing so connector lines stay horizontal.
    """
    total_matches = 16 >> round_idx          # 16, 8, 4, 2
    half = total_matches // 2
    t = tops(total_matches)
    if side == "left":
        return t[:half]
    else:
        return t[half:]

# ── card & match HTML generators ──────────────────────────────────────────

BEST8_SET = set(best8_thirds)

def slot_html(team, col, side, is_third=False):
    row = ts(team)
    pw  = row["p_winner"]
    sp  = row[col]
    bg  = win_color(pw)
    bd  = "right" if side == "left" else "left"
    # Orange accent border for best-3rd qualifiers
    accent = "#d4860a" if is_third else "#9aafa0"
    tag = '<span class="b3-tag">3rd ★</span>' if is_third else ""
    return (f'<div class="slot" style="background:{bg};border-{bd}:3px solid {accent};">'
            f'<span class="tn">{team}</span>'
            f'<span class="tp">{tag}{sp:.1f}%</span>'
            f'</div>')

def match_card(a, b, top, col, side, card_left=0, a_third=False, b_third=False):
    return (f'<div class="mc" style="top:{top:.1f}px;left:{card_left}px;width:{CW}px;">'
            f'{slot_html(a, col, side, a_third)}'
            f'<div class="sg"></div>'
            f'{slot_html(b, col, side, b_third)}'
            f'</div>')

def round_col_html(matches, round_idx, side):
    col = STAGE_COLS[round_idx]
    ts_ = global_tops(round_idx, side)
    card_left = CON if side == "right" else 0
    cards = "".join(
        match_card(a, b, ts_[i], col, side, card_left,
                   a_third=(round_idx == 0 and a in BEST8_SET),
                   b_third=(round_idx == 0 and b in BEST8_SET))
        for i, (a, b) in enumerate(matches)
    )
    return (f'<div class="rc" style="width:{RW}px;height:{BH}px;position:relative;">'
            f'{cards}</div>')

# ── SVG connectors (single overlay) ───────────────────────────────────────

def make_svg():
    lines = []

    def h(x1, y1, x2, y2):
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')

    # X offsets of each column's LEFT edge in the bracket
    col_x = {
        "r32l": 0,
        "r16l": RW,
        "qfl":  2*RW,
        "sfl":  3*RW,
        "fin":  4*RW,
        "sfr":  4*RW + FW,
        "qfr":  4*RW + FW + RW,
        "r16r": 4*RW + FW + 2*RW,
        "r32r": 4*RW + FW + 3*RW,
    }

    # For left side: cards at x=col_x[name]+0, connector exits rightward from CARD_W
    # For right side: cards at x=col_x[name]+CON, connector exits leftward from col_x[name]+CON

    # Left side connectors: R32→R16→QF→SF
    left_names = ["r32l", "r16l", "qfl", "sfl"]
    for ri in range(3):  # connect round ri to ri+1
        src_name, dst_name = left_names[ri], left_names[ri+1]
        src_ts = global_tops(ri, "left")
        dst_ts = global_tops(ri+1, "left")
        x_src_right = col_x[src_name] + CW
        x_vert = col_x[src_name] + CW + CON * 0.6
        x_dst_left  = col_x[dst_name]  # = col_x[src_name] + RW = left edge of next card

        for i, ty in enumerate(src_ts):
            y = ty + MATCH_H / 2
            h(x_src_right, y, x_vert, y)          # exit stub →

        for pair in range(len(src_ts) // 2):
            y0 = src_ts[pair*2]   + MATCH_H / 2
            y1 = src_ts[pair*2+1] + MATCH_H / 2
            ym = dst_ts[pair]     + MATCH_H / 2
            h(x_vert, y0, x_vert, y1)              # vertical connector
            h(x_vert, ym, x_dst_left, ym)          # junction →

    # SF-L → Final
    sf_l_ts = global_tops(3, "left")
    y_sfl = sf_l_ts[0] + MATCH_H / 2
    h(col_x["sfl"] + CW, y_sfl, col_x["fin"], y_sfl)

    # Right side connectors: R32→R16→QF→SF (but rendered right-to-left)
    right_names = ["r32r", "r16r", "qfr", "sfr"]
    for ri in range(3):
        src_name, dst_name = right_names[ri], right_names[ri+1]
        src_ts = global_tops(ri, "right")
        dst_ts = global_tops(ri+1, "right")
        x_src_left = col_x[src_name] + CON         # left edge of card (connector space on left)
        x_vert = col_x[src_name] + CON * 0.4       # vertical placed in connector space
        x_dst_right = col_x[src_name]              # = right edge of the "closer to center" column's card

        for ty in src_ts:
            y = ty + MATCH_H / 2
            h(x_src_left, y, x_vert, y)            # exit stub ←

        for pair in range(len(src_ts) // 2):
            y0 = src_ts[pair*2]   + MATCH_H / 2
            y1 = src_ts[pair*2+1] + MATCH_H / 2
            ym = dst_ts[pair]     + MATCH_H / 2
            h(x_vert, y0, x_vert, y1)              # vertical connector
            h(x_vert, ym, x_dst_right, ym)         # junction ←

    # SF-R → Final (line from Final right edge to SF-R card left edge)
    sf_r_ts = global_tops(3, "right")
    y_sfr = sf_r_ts[0] + MATCH_H / 2
    h(col_x["fin"] + FW, y_sfr, col_x["sfr"] + CON, y_sfr)

    # Vertical in Final column connecting the two SF lines
    x_fin_mid = col_x["fin"] + FW / 2
    h(x_fin_mid, y_sfl, x_fin_mid, y_sfr)

    svg = (f'<svg style="position:absolute;top:0;left:0;width:{TOTAL_W}px;height:{BH}px;pointer-events:none">'
           f'<g stroke="#b0bec8" stroke-width="1.5" fill="none">'
           + "\n".join(lines) +
           f'</g></svg>')
    return svg

# ── final column HTML ──────────────────────────────────────────────────────

def final_col_html():
    sf_l_ts = global_tops(3, "left")
    sf_r_ts = global_tops(3, "right")
    y_l = sf_l_ts[0]
    y_r = sf_r_ts[0]
    cy  = (y_l + y_r) / 2  # champion label vertical center

    def fin_slot(team, y_pos, side):
        row = ts(team)
        bg  = win_color(row["p_winner"])
        bd  = "right" if side == "left" else "left"
        return (f'<div class="mc fin-mc" style="top:{y_pos:.1f}px;width:{FW-4}px;left:2px;">'
                f'<div class="slot" style="background:{bg};border-{bd}:3px solid #7aaa90;">'
                f'<span class="tn">{team}</span>'
                f'<span class="tp">{row["p_final"]:.1f}% final</span>'
                f'</div>'
                f'<div class="sg"></div>'
                f'<div class="slot blank"><span class="tn" style="color:#aaa">vs</span></div>'
                f'</div>')

    # Champion badge
    ch = ts(CHAMP)
    ch_y = cy - 28
    badge = (f'<div style="position:absolute;top:{ch_y:.1f}px;left:0;width:{FW}px;text-align:center;z-index:5">'
             f'<div class="champ-badge" style="background:{win_color(ch["p_winner"])}">'
             f'🏆 <strong>{CHAMP}</strong><br>'
             f'<span style="font-size:.72rem">{ch["p_winner"]:.2f}% win prob</span>'
             f'</div></div>')

    return (f'<div class="rc fin-rc" style="width:{FW}px;height:{BH}px;position:relative;">'
            f'{fin_slot(FIN_L, y_l, "left")}'
            f'{fin_slot(FIN_R, y_r, "right")}'
            f'{badge}'
            f'</div>')

# ── bracket HTML ───────────────────────────────────────────────────────────

def bracket_html():
    # Left side: R32 → SF (left to right)
    left_cols = "".join(round_col_html(L_ROUNDS[i], i, "left") for i in range(4))
    # Right side: SF → R32 (left to right, so we render SF first then outward)
    # But with global_tops(side="right"), positions are already in the bottom half
    right_cols = "".join(round_col_html(R_ROUNDS[i], i, "right") for i in range(3, -1, -1))

    # Column labels
    def lbl(text, w):
        return f'<div style="width:{w}px;text-align:center;font-size:.7rem;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:.06em;flex-shrink:0">{text}</div>'

    labels = (lbl("R32", RW) + lbl("R16", RW) + lbl("QF", RW) + lbl("SF", RW) +
              lbl("Final", FW) +
              lbl("SF", RW) + lbl("QF", RW) + lbl("R16", RW) + lbl("R32", RW))

    return f"""
<div style="overflow-x:auto">
  <div style="display:flex;margin-bottom:.4rem;width:{TOTAL_W}px">{labels}</div>
  <div style="position:relative;width:{TOTAL_W}px;height:{BH}px;display:flex;">
    {left_cols}
    {final_col_html()}
    {right_cols}
    {make_svg()}
  </div>
</div>"""

# ── group stage HTML ───────────────────────────────────────────────────────

GRP_COLS = [("p_1st","1st"),("p_2nd","2nd"),("p_3rd","3rd"),("p_4th","4th")]

def cell_bg(v):
    if v < 5:   return ""
    t = min(v / 70, 1.0)
    r, g, b = int(240 - 40*t), int(252 - 30*t), int(235 - 40*t)
    return f"background:rgb({r},{g},{b})"

def third_bg(v):
    """Orange-tinted background for Best 3rd column."""
    if v < 1: return ""
    t = min(v / 25, 1.0)
    r, g, b = int(255 - 10*t), int(240 - 60*t), int(210 - 50*t)
    return f"background:rgb({r},{g},{b})"

def group_html(name):
    rows = ""
    for _, r in df[df["group"] == name].sort_values("p_1st", ascending=False).iterrows():
        cells = f'<td class="gtn">{r["team"]}</td>'
        for col, _ in GRP_COLS:
            bg = cell_bg(r[col])
            s  = f' style="{bg}"' if bg else ""
            cells += f'<td{s}>{r[col]:.1f}%</td>'
        # Top-2 qualify probability (guaranteed path)
        p_top2 = r["p_1st"] + r["p_2nd"]
        bg2 = cell_bg(p_top2)
        s2  = f' style="{bg2}"' if bg2 else ""
        cells += f'<td{s2}>{p_top2:.1f}%</td>'
        # Best-3rd qualify probability (competitive path)
        p_b3 = max(0.0, r["p_qualified"] - p_top2)
        bg3 = third_bg(p_b3)
        s3  = f' style="{bg3}"' if bg3 else ""
        cells += f'<td{s3}>{p_b3:.1f}%</td>'
        rows += f"<tr>{cells}</tr>"
    hdr = "".join(f"<th>{lbl}</th>" for _, lbl in GRP_COLS)
    hdr += '<th class="th-top2">Via Top 2</th><th class="th-b3">Best 3rd ★</th>'
    return f"""<div class="gc">
<div class="gh">Group {name}</div>
<table><thead><tr><th></th>{hdr}</tr></thead><tbody>{rows}</tbody></table>
</div>"""

groups_html = "\n".join(group_html(g) for g in GS)

# ── assemble page ──────────────────────────────────────────────────────────

PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>2026 FIFA World Cup Predictions</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; background: #eef0f5; color: #1a1a2e; }}
header {{ background: #1a1a2e; color: #fff; padding: 1.3rem 2rem; }}
header h1 {{ font-size: 1.45rem; }}
header p  {{ font-size: .82rem; opacity: .65; margin-top: .25rem; }}
.sec      {{ padding: 1.8rem 2rem; }}
.stitle   {{ font-size: 1.1rem; font-weight: 700; border-left: 4px solid #6a9cbf;
             padding-left: .65rem; margin-bottom: 1.1rem; }}

/* group grid */
.ggrid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .85rem; }}
.gc   {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow: hidden; }}
.gh   {{ background: #1a1a2e; color: #fff; padding: .38rem .8rem; font-weight: 700; font-size: .88rem; }}
.gc table {{ width: 100%; border-collapse: collapse; font-size: .8rem; }}
.gc th {{ background: #e8eaf6; padding: .32rem .45rem; text-align: center;
          font-weight: 600; font-size: .75rem; color: #333; }}
.gc th:first-child {{ text-align: left; padding-left: .75rem; }}
.gc td {{ padding: .32rem .45rem; text-align: center; border-top: 1px solid #f2f2f2; }}
td.gtn  {{ text-align: left; padding-left: .75rem; font-weight: 500; }}
.gc tr:hover td {{ background: #f4f7ff !important; }}

/* bracket */
.bwrap {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
          padding: 1rem 1.2rem 1.4rem; }}
.rc   {{ flex-shrink: 0; }}
.fin-rc {{ border-left: 2px dashed #c5cfe0; border-right: 2px dashed #c5cfe0; }}
.mc   {{ position: absolute; border-radius: 5px; overflow: hidden;
         box-shadow: 0 1px 5px rgba(0,0,0,.13); }}
.fin-mc {{ box-shadow: 0 2px 8px rgba(0,0,0,.18); }}
.slot {{ height: {SLOT}px; display: flex; align-items: center;
         justify-content: space-between; padding: 0 6px; font-size: .75rem; }}
.slot.blank {{ background: #f5f6f8; }}
.sg   {{ height: {GAP}px; background: #cdd3dc; }}
.tn   {{ font-weight: 600; white-space: nowrap; overflow: hidden;
         text-overflow: ellipsis; max-width: 90px; }}
.tp   {{ font-size: .68rem; color: #445; white-space: nowrap; padding-left: 4px; }}
.champ-badge {{ display: inline-block; border-radius: 8px; padding: .45rem .8rem;
                font-size: .82rem; line-height: 1.5; box-shadow: 0 2px 8px rgba(0,0,0,.18);
                border: 2px solid #6a9 }}
/* best-3rd accents */
.th-top2 {{ background: #e8f4e8 !important; }}
.th-b3   {{ background: #fff3e0 !important; color: #a05a00 !important; }}
.b3-tag  {{ font-size: .6rem; background: #d4860a; color: #fff; border-radius: 3px;
            padding: 0 3px; margin-right: 3px; font-weight: 700; }}
/* format note */
.fmt-note {{ display:flex; gap:1.2rem; flex-wrap:wrap; font-size:.8rem; color:#555;
             background:#fff; border-radius:8px; padding:.7rem 1rem;
             box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:1rem;
             border-left:4px solid #d4860a; }}
.fmt-note span {{ white-space:nowrap; }}
</style>
</head>
<body>
<header>
  <h1>2026 FIFA World Cup — Predicted Probabilities</h1>
  <p>XGBoost model trained on historical international results &nbsp;·&nbsp; 50,000 Monte Carlo simulations</p>
</header>

<div class="sec">
  <div class="stitle">Group Stage</div>
  <div class="fmt-note">
    <span>📋 <strong>Format:</strong> 48 teams · 12 groups of 4</span>
    <span>✅ <strong>Top 2</strong> from every group qualify automatically (24 teams)</span>
    <span>⭐ <strong>Best 8 third-place</strong> teams also advance (8 teams)</span>
    <span>→ 32 teams enter the Round of 32</span>
    <span style="color:#a05a00">Orange ★ = probability of qualifying as one of the best 8 thirds</span>
  </div>
  <div class="ggrid">{groups_html}</div>
</div>

<div class="sec">
  <div class="stitle">Knockout Bracket — Expected Path</div>
  <p style="font-size:.8rem;color:#666;margin-bottom:.9rem">
    Each slot shows the most likely team to advance to that round.
    Numbers = probability of reaching that round. Color intensity = tournament win probability.
    <span style="color:#a05a00;font-weight:600">Orange border + ★ = best-3rd qualifier slot.</span>
  </p>
  <div class="bwrap">{bracket_html()}</div>
</div>
</body>
</html>"""

out = Path("data/predictions.html")
out.write_text(PAGE)
print(f"Saved → {out.resolve()}")
