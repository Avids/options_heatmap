import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime

st.set_page_config(page_title="Options Net Heatmap (Static)", layout="wide")

# =============================
# HELPERS
# =============================
def format_oi_value(val):
    try:
        v = int(val)
    except Exception:
        return ""
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,}"

def fetch_ticker(ticker_symbol: str):
    return yf.Ticker(ticker_symbol)

@st.cache_data(show_spinner=False)
def get_price(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    hist = t.history(period="1d")
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])

def get_option_chain_for_expiry(ticker, expiry):
    try:
        chain = ticker.option_chain(expiry)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return None, None

# =============================
# SIDEBAR / INPUTS
# =============================
st.title("Options Net Heatmap (static matplotlib)")
st.markdown("Visualize Net Open Interest or Net Volume (Calls − Puts) across actual strikes and expiries using a static chart.")

col_inputs, col_info = st.columns([1, 2])

with col_inputs:
    symbol = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    METRIC = st.selectbox("Metric", options=["Open Interest", "Volume"])
    STRIKE_RANGE = st.number_input(
        "Strikes above / below current price (count)",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )
    EXPIRY_COUNT = st.slider(
        "Number of expiries to include", min_value=1, max_value=24, value=4
    )
    refresh = st.button("Update (re-fetch)")

with col_info:
    st.markdown("Usage tips:")
    st.markdown("- Choose metric: Open Interest (openInterest) or Volume (volume).")
    st.markdown("- Displays actual strike levels (no binning).")
    st.markdown("- Choose how many strikes above and below current price to show.")
    st.markdown(f"- Report generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC")

if not symbol:
    st.info("Enter a ticker symbol to continue.")
    st.stop()

metric_col = "openInterest" if METRIC == "Open Interest" else "volume"
colorbar_title = f"Net {METRIC} (Calls − Puts)"

# =============================
# FETCH DATA
# =============================
with st.spinner(f"Fetching data for {symbol} ..."):
    price = get_price(symbol)
    ticker = fetch_ticker(symbol)
    try:
        all_expiries = list(ticker.options)
    except Exception:
        all_expiries = []

if price is None:
    st.error(f"Could not fetch price or options for '{symbol}'. Check the ticker and try again.")
    st.stop()

if not all_expiries:
    st.error(f"No option expiries returned for '{symbol}'.")
    st.stop()

expiries = all_expiries[:EXPIRY_COUNT]

all_data = []
missing_metric_warned = False

for expiry in expiries:
    calls, puts = get_option_chain_for_expiry(ticker, expiry)
    if calls is None or puts is None:
        continue

    if "strike" not in calls.columns or "strike" not in puts.columns:
        continue

    # If metric column missing, fill with zeros and warn once
    if metric_col not in calls.columns:
        calls[metric_col] = 0
        if not missing_metric_warned and METRIC == "Volume":
            st.warning("Volume column not present on some option chains; treating missing values as 0.")
            missing_metric_warned = True
    if metric_col not in puts.columns:
        puts[metric_col] = 0
        if not missing_metric_warned and METRIC == "Volume":
            st.warning("Volume column not present on some option chains; treating missing values as 0.")
            missing_metric_warned = True

    call_vals = calls.groupby("strike")[metric_col].sum()
    put_vals = puts.groupby("strike")[metric_col].sum()
    net = call_vals.subtract(put_vals, fill_value=0)

    df = net.reset_index()
    df.columns = ["strike", "net_metric"]
    df["expiry"] = expiry

    all_data.append(df)

if not all_data:
    st.error("No options data could be processed for the requested expiries.")
    st.stop()

nodes = pd.concat(all_data, ignore_index=True)

# =============================
# SELECT STRIKES AROUND PRICE (actual strikes)
# =============================
unique_strikes = np.sort(nodes["strike"].unique())
if unique_strikes.size == 0:
    st.error("No strikes found in the option chains.")
    st.stop()

closest_idx = int(np.abs(unique_strikes - price).argmin())
low_idx = max(0, closest_idx - STRIKE_RANGE)
high_idx = min(len(unique_strikes) - 1, closest_idx + STRIKE_RANGE)
selected_strikes = unique_strikes[low_idx : high_idx + 1]

nodes = nodes[nodes["strike"].isin(selected_strikes)].copy()
if nodes.empty:
    st.error("No nodes after selecting strikes around price. Try expanding the strike range.")
    st.stop()

# =============================
# BUILD HEATMAP (sort descending so top is highest strike)
# =============================
heatmap = nodes.pivot_table(index="strike", columns="expiry", values="net_metric", aggfunc="sum").fillna(0)

# sort descending -> highest strike first row (will display at top with origin='upper')
heatmap = heatmap.sort_index(ascending=False)

expiry_labels = [pd.to_datetime(x).date().isoformat() for x in heatmap.columns]
strike_labels = [str(int(s)) for s in heatmap.index]  # highest -> lowest

z = heatmap.values  # rows = strike_labels order highest->lowest

# =============================
# FIND KING CALL & KING PUT NODES
# =============================
king_call = {}
king_put = {}
for col in heatmap.columns:
    col_series = heatmap[col]
    if col_series.size == 0:
        continue
    king_call[col] = int(col_series.idxmax())
    king_put[col] = int(col_series.idxmin())

# =============================
# PLOT (matplotlib static)
# =============================
fig, ax = plt.subplots(figsize=(12, max(6, len(strike_labels) * 0.18)))

# colormap and normalization centered at zero
cmap = plt.get_cmap("RdYlGn")
# symmetric norm around 0 for good divergence coloring
max_abs = np.nanmax(np.abs(z)) if z.size else 1
norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

im = ax.imshow(z, aspect="auto", cmap=cmap, norm=norm, origin="upper")

# ticks
ax.set_xticks(np.arange(len(expiry_labels)))
ax.set_xticklabels(expiry_labels, rotation=30, ha="right")
ax.set_yticks(np.arange(len(strike_labels)))
ax.set_yticklabels(strike_labels)

ax.set_xlabel("Expiry (date)")
ax.set_ylabel("Strike")
ax.set_title(f"{symbol} | Price: {price:.2f} | Net {METRIC} Heatmap (King Call & Put Nodes)")

# add colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label(colorbar_title)

# annotations (text) and king highlights
# choose threshold for text color contrast
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        val = z[i, j]
        txt = format_oi_value(val)
        # find if this cell is king call or king put
        strike_val = int(heatmap.index[i])
        expiry_val = heatmap.columns[j]
        is_king = (expiry_val in king_call and strike_val == king_call[expiry_val]) or (
            expiry_val in king_put and strike_val == king_put[expiry_val]
        )

        # text color depends on background intensity
        text_color = "white" if abs(val) > (max_abs * 0.35) else "black"
        fontweight = "bold" if is_king else "normal"
        fontsize = 10 if is_king else 8

        ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=fontsize, fontweight=fontweight)

        # draw a rectangle around king cells
        if is_king:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

plt.tight_layout()

# display static chart in Streamlit
st.pyplot(fig)

# =============================
# SUMMARY & DOWNLOAD
# =============================
st.subheader(f"{symbol}  •  Price: {price:.2f}")
st.markdown(f"Metric: **{METRIC}** — Showing {len(strike_labels)} strike levels and {len(expiry_labels)} expiries")

with st.expander("King nodes (by expiry)"):
    rows = []
    for expiry in heatmap.columns:
        rows.append(
            f"- {pd.to_datetime(expiry).date().isoformat()}: King Call = {king_call.get(expiry)}, King Put = {king_put.get(expiry)}"
        )
    st.markdown("\n".join(rows))

csv = nodes[["expiry", "strike", "net_metric"]].sort_values(["expiry", "strike"])
csv = csv.rename(columns={"net_metric": f"net_{METRIC.replace(' ', '_').lower()}"})
csv_bytes = csv.to_csv(index=False).encode("utf-8")
st.download_button(
    f"Download CSV of net {METRIC} by strike",
    data=csv_bytes,
    file_name=f"{symbol}_net_{METRIC.replace(' ', '_').lower()}_{datetime.utcnow().date()}.csv",
    mime="text/csv",
)
