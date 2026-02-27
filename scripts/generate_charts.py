"""
Baku Rental Market — Chart Generator
Produces all business insight charts from data/data.csv → charts/
"""

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE  = Path(__file__).parent.parent / "data" / "data.csv"
CHARTS_DIR = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
BRAND_BLUE    = "#1A3A5C"
BRAND_TEAL    = "#2E86AB"
BRAND_ORANGE  = "#F4A261"
BRAND_GREEN   = "#2A9D8F"
BRAND_LIGHT   = "#E8F4F8"
ACCENT_RED    = "#E76F51"

PALETTE = [BRAND_BLUE, BRAND_TEAL, BRAND_GREEN, BRAND_ORANGE, ACCENT_RED,
           "#457B9D", "#A8DADC", "#6A0572", "#E9C46A", "#264653"]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#E0E0E0",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "axes.axisbelow":   True,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

def flt(v):
    try:    return float(v)
    except: return None

def add_value_labels(ax, bars, fmt="{:.0f}", offset=3):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=9, color="#333333", fontweight="bold",
            )

def save(fig, name):
    path = CHARTS_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {name}")

# ── Load data ──────────────────────────────────────────────────────────────────
with open(DATA_FILE, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows):,} listings\n")

# Clean district aliases (encoding artefacts)
_alias = {"NÉsimi": "Nəsimi", "XÉtai": "Xətai", "NÉrimanov": "Nərimanov"}
for r in rows:
    r["district"] = _alias.get(r["district"], r["district"])

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Rental Price Distribution
# ─────────────────────────────────────────────────────────────────────────────
buckets = [0, 300, 500, 700, 1_000, 1_500, 2_000, 3_000, 1e9]
labels  = ["<300", "300–500", "500–700", "700–1K", "1K–1.5K", "1.5K–2K", "2K–3K", ">3K"]
counts  = [0] * 8
for r in rows:
    p = flt(r["price_azn"])
    if p:
        for i in range(len(buckets) - 1):
            if buckets[i] <= p < buckets[i + 1]:
                counts[i] += 1
                break

fig, ax = plt.subplots(figsize=(11, 5))
colors = [BRAND_TEAL if "700" in l or "1K" in l else BRAND_BLUE for l in labels]
bars = ax.bar(labels, counts, color=colors, width=0.65, zorder=3)
add_value_labels(ax, bars, offset=8)
ax.set_title("Rental Price Distribution — Monthly Rent (AZN)", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Monthly Rent Range (AZN)", fontsize=11)
ax.set_ylabel("Number of Listings", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
# annotate dominant segment
ax.annotate("51% of all\nlistings here",
    xy=(3.5, max(counts[3], counts[4]) / 2),
    xytext=(5.5, max(counts) * 0.75),
    arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.5),
    fontsize=9, color=ACCENT_RED, fontweight="bold")
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "01_price_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Listings by District (Supply)
# ─────────────────────────────────────────────────────────────────────────────
CORE_DISTRICTS = ["Yasamal", "Nəsimi", "Nərimanov", "Xətai",
                  "Nizami", "Səbail", "Binəqədi", "Abşeron"]
dist_count = Counter(r["district"] for r in rows if r["district"])
dist_vals  = [dist_count.get(d, 0) for d in CORE_DISTRICTS]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(CORE_DISTRICTS, dist_vals,
              color=[BRAND_BLUE if v > 300 else BRAND_TEAL for v in dist_vals],
              width=0.6, zorder=3)
add_value_labels(ax, bars, offset=6)
ax.set_title("Rental Supply by District — Number of Active Listings", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Active Listings", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
total = sum(dist_vals)
ax.text(0.99, 0.95,
    f"Top 4 districts = {sum(dist_vals[:4])/total*100:.0f}% of supply",
    transform=ax.transAxes, ha="right", fontsize=9,
    bbox=dict(facecolor=BRAND_LIGHT, edgecolor=BRAND_TEAL, boxstyle="round,pad=0.4"))
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "02_listings_by_district.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Median Monthly Rent by District
# ─────────────────────────────────────────────────────────────────────────────
by_dist_price = defaultdict(list)
for r in rows:
    d = r["district"]
    p = flt(r["price_azn"])
    if d in CORE_DISTRICTS and p and p < 10_000:
        by_dist_price[d].append(p)

median_by_dist = {d: round(statistics.median(by_dist_price[d])) for d in CORE_DISTRICTS}
sorted_dist    = sorted(CORE_DISTRICTS, key=lambda d: -median_by_dist[d])
sorted_vals    = [median_by_dist[d] for d in sorted_dist]
overall_median = round(statistics.median([flt(r["price_azn"]) for r in rows if flt(r["price_azn"])]))

fig, ax = plt.subplots(figsize=(11, 5))
bar_colors = [BRAND_ORANGE if v > overall_median else BRAND_TEAL for v in sorted_vals]
bars = ax.bar(sorted_dist, sorted_vals, color=bar_colors, width=0.6, zorder=3)
add_value_labels(ax, bars, fmt="{:.0f} AZN", offset=8)
ax.axhline(overall_median, color=ACCENT_RED, linewidth=1.8, linestyle="--", zorder=4)
ax.text(len(sorted_dist) - 0.5, overall_median + 15,
        f"Market median: {overall_median} AZN", color=ACCENT_RED, fontsize=9, fontweight="bold")
ax.set_title("Median Monthly Rent by District (AZN)", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Median Rent (AZN / month)", fontsize=11)
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "03_median_price_by_district.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Price per m² by District
# ─────────────────────────────────────────────────────────────────────────────
by_dist_pm2 = defaultdict(list)
for r in rows:
    d = r["district"]
    pm2 = flt(r["price_per_m2"])
    if d in CORE_DISTRICTS and pm2 and pm2 < 100:
        by_dist_pm2[d].append(pm2)

median_pm2 = {d: round(statistics.median(by_dist_pm2[d]), 1) for d in CORE_DISTRICTS}
sorted_pm2_dist = sorted(CORE_DISTRICTS, key=lambda d: -median_pm2[d])
sorted_pm2_vals = [median_pm2[d] for d in sorted_pm2_dist]
overall_pm2 = round(statistics.median([flt(r["price_per_m2"]) for r in rows if flt(r["price_per_m2"]) and flt(r["price_per_m2"]) < 100]), 1)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(sorted_pm2_dist, sorted_pm2_vals,
              color=[BRAND_ORANGE if v > overall_pm2 else BRAND_TEAL for v in sorted_pm2_vals],
              width=0.6, zorder=3)
add_value_labels(ax, bars, fmt="{:.1f}", offset=0.2)
ax.axhline(overall_pm2, color=ACCENT_RED, linewidth=1.8, linestyle="--", zorder=4)
ax.text(len(sorted_pm2_dist) - 0.5, overall_pm2 + 0.2,
        f"Market median: {overall_pm2} AZN/m²", color=ACCENT_RED, fontsize=9, fontweight="bold")
ax.set_title("Median Rent per m² by District (AZN/m²)", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Median Rent per m² (AZN)", fontsize=11)
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "04_price_per_m2_by_district.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Listings by Room Count
# ─────────────────────────────────────────────────────────────────────────────
ROOM_LABELS = ["1-bed", "2-bed", "3-bed", "4-bed", "5-bed"]
room_filter = {1: "1-bed", 2: "2-bed", 3: "3-bed", 4: "4-bed", 5: "5-bed"}
room_counts = Counter()
for r in rows:
    rm = flt(r["rooms"])
    if rm and int(rm) in room_filter:
        room_counts[room_filter[int(rm)]] += 1
room_vals = [room_counts[l] for l in ROOM_LABELS]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(ROOM_LABELS, room_vals,
              color=[BRAND_BLUE if l == "2-bed" else BRAND_TEAL for l in ROOM_LABELS],
              width=0.55, zorder=3)
add_value_labels(ax, bars, offset=8)
total_rm = sum(room_vals)
for bar, val in zip(bars, room_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, 30,
            f"{val/total_rm*100:.0f}%",
            ha="center", fontsize=9, color="white", fontweight="bold")
ax.set_title("Market Supply by Apartment Size", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Number of Listings", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.text(0.01, 0.95, "2-bed dominates at 50% of supply",
        transform=ax.transAxes, fontsize=9, color=BRAND_BLUE, fontweight="bold",
        bbox=dict(facecolor=BRAND_LIGHT, edgecolor=BRAND_BLUE, boxstyle="round,pad=0.3"))
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "05_listings_by_rooms.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 — Median Price & Area by Room Count (dual metric)
# ─────────────────────────────────────────────────────────────────────────────
by_rooms_price = defaultdict(list)
by_rooms_area  = defaultdict(list)
for r in rows:
    rm = flt(r["rooms"])
    p  = flt(r["price_azn"])
    a  = flt(r["area_m2"])
    if rm and int(rm) in range(1, 6):
        if p and p < 10_000: by_rooms_price[int(rm)].append(p)
        if a and a < 500:    by_rooms_area[int(rm)].append(a)

rm_labels = ["1-bed", "2-bed", "3-bed", "4-bed", "5-bed"]
rm_prices  = [round(statistics.median(by_rooms_price[i])) for i in range(1, 6)]
rm_areas   = [round(statistics.median(by_rooms_area[i]))  for i in range(1, 6)]

fig, ax1 = plt.subplots(figsize=(11, 5))
x = np.arange(len(rm_labels))
w = 0.38
bars1 = ax1.bar(x - w/2, rm_prices, w, color=BRAND_BLUE,  label="Median Rent (AZN)", zorder=3)
ax2   = ax1.twinx()
bars2 = ax2.bar(x + w/2, rm_areas,  w, color=BRAND_ORANGE, label="Median Area (m²)",  zorder=3)
add_value_labels(ax1, bars1, fmt="{:.0f} AZN", offset=10)
add_value_labels(ax2, bars2, fmt="{:.0f} m²", offset=1)
ax1.set_xticks(x); ax1.set_xticklabels(rm_labels)
ax1.set_ylabel("Median Rent (AZN)", color=BRAND_BLUE, fontsize=11)
ax2.set_ylabel("Median Area (m²)", color=BRAND_ORANGE, fontsize=11)
ax1.tick_params(axis="y", labelcolor=BRAND_BLUE)
ax2.tick_params(axis="y", labelcolor=BRAND_ORANGE)
ax1.spines["right"].set_visible(False)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(BRAND_ORANGE)
lines = [plt.Rectangle((0,0),1,1,color=BRAND_BLUE), plt.Rectangle((0,0),1,1,color=BRAND_ORANGE)]
ax1.legend(lines, ["Median Rent (AZN)", "Median Area (m²)"], loc="upper left", fontsize=9)
ax1.set_title("Median Rent & Apartment Size by Room Count", fontsize=14, fontweight="bold", pad=14)
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "06_price_by_rooms.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 7 — New vs Old Buildings: Count & Price Comparison
# ─────────────────────────────────────────────────────────────────────────────
new_label = "New Build\n(yeni tikili)"
old_label = "Old Build\n(köhnə tikili)"
new_prices_l = [flt(r["price_azn"]) for r in rows if r.get("building_type") == "yeni tikili" and flt(r.get("price_azn"))]
old_prices_l = [flt(r["price_azn"]) for r in rows if r.get("building_type") == "köhnə tikili" and flt(r.get("price_azn"))]

fig, (ax_cnt, ax_prc) = plt.subplots(1, 2, figsize=(13, 5))
# Count
cnt_bars = ax_cnt.bar([new_label, old_label], [len(new_prices_l), len(old_prices_l)],
                      color=[BRAND_BLUE, BRAND_TEAL], width=0.45, zorder=3)
add_value_labels(ax_cnt, cnt_bars, offset=10)
for bar, val in zip(cnt_bars, [len(new_prices_l), len(old_prices_l)]):
    total_bt = len(new_prices_l) + len(old_prices_l)
    ax_cnt.text(bar.get_x() + bar.get_width()/2, 30,
                f"{val/total_bt*100:.0f}%", ha="center", fontsize=11, color="white", fontweight="bold")
ax_cnt.set_title("Supply Split:\nNew vs Old Construction", fontsize=12, fontweight="bold")
ax_cnt.set_ylabel("Number of Listings", fontsize=10)
# Price
new_med = round(statistics.median(new_prices_l))
old_med = round(statistics.median(old_prices_l))
prc_bars = ax_prc.bar([new_label, old_label], [new_med, old_med],
                      color=[BRAND_BLUE, BRAND_TEAL], width=0.45, zorder=3)
add_value_labels(ax_prc, prc_bars, fmt="{:.0f} AZN", offset=8)
premium = round((new_med - old_med) / old_med * 100)
ax_prc.annotate(f"+{premium}% premium",
    xy=(0, new_med), xytext=(0.8, new_med + 60),
    arrowprops=dict(arrowstyle="->", color=ACCENT_RED),
    fontsize=10, color=ACCENT_RED, fontweight="bold")
ax_prc.set_title("Median Monthly Rent:\nNew vs Old Construction", fontsize=12, fontweight="bold")
ax_prc.set_ylabel("Median Rent (AZN / month)", fontsize=10)
fig.suptitle("New vs Old Buildings — Supply & Pricing", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.text(0.99, -0.02, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "07_new_vs_old_buildings.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 8 — Floor Position vs Median Rent
# ─────────────────────────────────────────────────────────────────────────────
floor_cats = defaultdict(list)
for r in rows:
    fl = flt(r["floor"]); tf = flt(r["total_floors"]); p = flt(r["price_azn"])
    if fl and tf and p and p < 10_000:
        if fl == 1:
            cat = "Ground Floor"
        elif fl == tf:
            cat = "Top Floor"
        elif fl / tf <= 0.33:
            cat = "Lower Floors\n(1st–3rd)"
        elif fl / tf <= 0.66:
            cat = "Middle Floors"
        else:
            cat = "Upper Floors"
        floor_cats[cat].append(p)

cat_order = ["Ground Floor", "Lower Floors\n(1st–3rd)", "Middle Floors", "Upper Floors", "Top Floor"]
cat_vals   = [round(statistics.median(floor_cats[c])) if floor_cats[c] else 0 for c in cat_order]
cat_sizes  = [len(floor_cats[c]) for c in cat_order]

fig, ax = plt.subplots(figsize=(11, 5))
bar_colors = [BRAND_ORANGE if v < 750 else BRAND_BLUE for v in cat_vals]
bars = ax.bar(cat_order, cat_vals, color=bar_colors, width=0.55, zorder=3)
add_value_labels(ax, bars, fmt="{:.0f} AZN", offset=8)
for bar, sz in zip(bars, cat_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, 30,
            f"n={sz:,}", ha="center", fontsize=8, color="white", fontweight="bold")
ax.set_title("Median Rent by Floor Position", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Median Rent (AZN / month)", fontsize=11)
ax.set_ylim(0, max(cat_vals) * 1.2)
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "08_price_by_floor_position.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 9 — Top Metro Station Hotspots
# ─────────────────────────────────────────────────────────────────────────────
metro_filter = lambda s: (
    s and s not in ("", "None") and "m." in s
    and "yaxınlığında" not in s
    and len(s) < 40
)
metro_counts = Counter(r["metro_station"] for r in rows if metro_filter(r.get("metro_station", "")))
top_metro    = metro_counts.most_common(12)
m_labels     = [m.replace(" m.", "") for m, _ in top_metro]
m_vals       = [v for _, v in top_metro]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(m_labels[::-1], m_vals[::-1],
               color=[BRAND_BLUE if v > 150 else BRAND_TEAL for v in m_vals[::-1]],
               height=0.6, zorder=3)
for bar, val in zip(bars, m_vals[::-1]):
    ax.text(val + 3, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9, fontweight="bold")
ax.set_title("Most Active Metro Station Catchment Areas\n(listings within walking distance)", fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Number of Listings", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "09_metro_station_hotspots.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 10 — Market Segments: Supply vs Median Rent (District Bubble)
# ─────────────────────────────────────────────────────────────────────────────
seg_labels = ["Budget\n(<500 AZN)", "Mid-Range\n(500–1K AZN)", "Upper-Mid\n(1K–2K AZN)", "Premium\n(>2K AZN)"]
seg_counts = [0, 0, 0, 0]
for r in rows:
    p = flt(r["price_azn"])
    if p:
        if p < 500:    seg_counts[0] += 1
        elif p < 1000: seg_counts[1] += 1
        elif p < 2000: seg_counts[2] += 1
        else:          seg_counts[3] += 1

seg_colors = [BRAND_GREEN, BRAND_BLUE, BRAND_TEAL, BRAND_ORANGE]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(seg_labels, seg_counts, color=seg_colors, width=0.55, zorder=3)
add_value_labels(ax, bars, offset=8)
total_seg = sum(seg_counts)
for bar, val in zip(bars, seg_counts):
    ax.text(bar.get_x() + bar.get_width()/2, 30,
            f"{val/total_seg*100:.0f}%", ha="center", fontsize=11, color="white", fontweight="bold")
ax.set_title("Baku Rental Market — Four Demand Segments", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Number of Listings", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.text(0.99, 0.95,
    f"Total listings: {total_seg:,}",
    transform=ax.transAxes, ha="right", fontsize=9,
    bbox=dict(facecolor=BRAND_LIGHT, edgecolor=BRAND_TEAL, boxstyle="round,pad=0.4"))
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "10_market_segments.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 11 — Supply vs Premium: District Opportunity Map
# ─────────────────────────────────────────────────────────────────────────────
opp_dists = ["Yasamal", "Nəsimi", "Nərimanov", "Xətai", "Nizami", "Səbail", "Binəqədi", "Abşeron"]
opp_supply = [dist_count.get(d, 0) for d in opp_dists]
opp_price  = [median_by_dist.get(d, 0) for d in opp_dists]

fig, ax = plt.subplots(figsize=(11, 6))
scatter_colors = [BRAND_ORANGE if p > overall_median else BRAND_TEAL for p in opp_price]
sc = ax.scatter(opp_supply, opp_price, s=[c/3 for c in opp_supply],
                c=scatter_colors, alpha=0.85, zorder=3, edgecolors="white", linewidths=1.5)
for d, x, y in zip(opp_dists, opp_supply, opp_price):
    ax.annotate(d, (x, y), textcoords="offset points", xytext=(8, 5),
                fontsize=10, fontweight="bold", color="#1A1A2E")
ax.axhline(overall_median, color=ACCENT_RED, linestyle="--", linewidth=1.5, alpha=0.7)
ax.axvline(sum(opp_supply)/len(opp_supply), color=BRAND_TEAL, linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(sum(opp_supply)/len(opp_supply) + 10, ax.get_ylim()[0] + 30,
        "Avg supply", fontsize=8, color=BRAND_TEAL)
ax.text(ax.get_xlim()[0] + 5, overall_median + 15,
        f"Market median {overall_median} AZN", fontsize=8, color=ACCENT_RED)
ax.set_title("District Opportunity Map — Supply Volume vs. Median Rent", fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Number of Active Listings (Supply Volume)", fontsize=11)
ax.set_ylabel("Median Monthly Rent (AZN)", fontsize=11)
# Quadrant labels
xlim = ax.get_xlim(); ylim = ax.get_ylim()
mid_x = sum(opp_supply)/len(opp_supply); mid_y = overall_median
ax.text(xlim[0]+10, ylim[1]-30, "Low supply\nHigh price →\nPremium niche", fontsize=7.5, color="gray", style="italic")
ax.text(xlim[1]-120, ylim[1]-30, "High supply\nHigh price →\nCore market", fontsize=7.5, color="gray", style="italic", ha="right")
ax.text(xlim[0]+10, ylim[0]+10, "Low supply\nLow price →\nEmerging", fontsize=7.5, color="gray", style="italic")
ax.text(xlim[1]-120, ylim[0]+10, "High supply\nLow price →\nCompetitive", fontsize=7.5, color="gray", style="italic", ha="right")
fig.text(0.99, 0.01, "Source: homdom.az · Feb 2026", ha="right", fontsize=8, color="gray")
save(fig, "11_district_opportunity_map.png")

print(f"\nAll charts saved to: {CHARTS_DIR}")
