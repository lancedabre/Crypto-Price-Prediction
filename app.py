import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ---------------------------------------------------------
# IMPORT TEAM MODULES
# ---------------------------------------------------------
from ml_pipeline import prepare_live_features
from nlp_pipeline import get_daily_sentiment

# ---------------------------------------------------------
# PAGE CONFIGURATION (Must be first)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Trading Terminal",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS — Dark terminal aesthetic with Bitcoin orange accents
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700;800&display=swap');

    /* Global */
    html, body, .stApp {
        background: #080c10 !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #1c2330 !important;
    }
    [data-testid="stSidebar"] * { font-family: 'Space Grotesk', sans-serif; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1117 0%, #111820 100%) !important;
        border: 1px solid #1c2330 !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        transition: border-color 0.25s, transform 0.2s, box-shadow 0.25s;
    }
    [data-testid="metric-container"]:hover {
        border-color: #f7931a44 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(247,147,26,0.08) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4e5d6c !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #e6edf3 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* Main title */
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        background: linear-gradient(135deg, #f7931a 0%, #ffcd3c 60%, #ff6b35 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #c9d1d9 !important;
        font-weight: 700 !important;
    }

    /* Dividers */
    hr {
        border: none !important;
        border-top: 1px solid #1c2330 !important;
        margin: 1.5rem 0 !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f7931a 0%, #e07b10 100%) !important;
        color: #080c10 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.8rem 2rem !important;
        box-shadow: 0 4px 20px rgba(247,147,26,0.35) !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 28px rgba(247,147,26,0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* Tabs */
    [data-testid="stTabs"] [role="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        color: #4e5d6c !important;
        border-bottom: 2px solid transparent !important;
        padding: 8px 16px !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #f7931a !important;
        border-bottom-color: #f7931a !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #0d1117 !important;
        border: 1px solid #1c2330 !important;
        border-radius: 10px !important;
    }

    /* Selectbox / Slider */
    [data-testid="stSelectbox"] > div, [data-testid="stSlider"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #080c10; }
    ::-webkit-scrollbar-thumb { background: #1c2330; border-radius: 2px; }

    /* Info card reusable */
    .card {
        background: linear-gradient(135deg, #0d1117 0%, #111820 100%);
        border: 1px solid #1c2330;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.75rem;
    }
    .card:hover { border-color: #1c2d3a; }

    .mono { font-family: 'JetBrains Mono', monospace; }

    /* Signal badge */
    .sig-bull {
        display:inline-block;background:rgba(34,197,94,0.12);
        border:1px solid rgba(34,197,94,0.4);color:#22c55e;
        padding:3px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;
        font-family:'JetBrains Mono',monospace;
    }
    .sig-bear {
        display:inline-block;background:rgba(239,68,68,0.12);
        border:1px solid rgba(239,68,68,0.4);color:#ef4444;
        padding:3px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;
        font-family:'JetBrains Mono',monospace;
    }
    .sig-neutral {
        display:inline-block;background:rgba(234,179,8,0.12);
        border:1px solid rgba(234,179,8,0.4);color:#eab308;
        padding:3px 12px;border-radius:20px;font-size:0.78rem;font-weight:700;
        font-family:'JetBrains Mono',monospace;
    }

    /* Status sidebar items */
    .sys-ok {
        background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.25);
        border-radius:8px;padding:9px 14px;margin-bottom:8px;
        color:#22c55e;font-size:0.82rem;display:flex;align-items:center;gap:8px;
    }
    .sys-fail {
        background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.25);
        border-radius:8px;padding:9px 14px;margin-bottom:8px;
        color:#ef4444;font-size:0.82rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TECHNICAL INDICATOR HELPERS
# ---------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def calculate_bollinger(series, period=20, std=2):
    sma = series.rolling(period).mean()
    dev = series.rolling(period).std()
    return sma + dev * std, sma, sma - dev * std

def max_drawdown(series):
    peak = series.cummax()
    return ((series - peak) / peak).min() * 100

def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=3)
        d = r.json()['data'][0]
        return int(d['value']), d['value_classification']
    except:
        return None, "N/A"

def signal_html(label, bull=False, bear=False):
    if bull:   return f'<span class="sig-bull">▲ {label}</span>'
    if bear:   return f'<span class="sig-bear">▼ {label}</span>'
    return f'<span class="sig-neutral">◆ {label}</span>'

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(8,12,16,0.95)',
    font=dict(color='#4e5d6c', family='JetBrains Mono, monospace', size=11),
    margin=dict(l=0, r=0, t=36, b=0),
)

def apply_grid(fig):
    """Apply consistent dark grid to all axes after layout is set."""
    fig.update_xaxes(gridcolor='#1c2330', showgrid=True)
    fig.update_yaxes(gridcolor='#1c2330', showgrid=True)


# ---------------------------------------------------------
# CACHED DATA LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_model.pkl')

@st.cache_data(ttl=300)
def fetch_data(ticker="BTC-USD", days=60):
    df = yf.download(ticker, period=f"{days}d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df['Close'].squeeze()
    df['SMA_7']     = close.rolling(7).mean()
    df['SMA_14']    = close.rolling(14).mean()
    df['SMA_50']    = close.rolling(50).mean()
    df['Daily_Return']   = close.pct_change()
    df['Volatility_7d']  = df['Daily_Return'].rolling(7).std()
    df['Cum_Return']     = (1 + df['Daily_Return']).cumprod() - 1
    df['RSI']       = calculate_rsi(close)
    df['MACD'], df['MACD_Sig'], df['MACD_Hist'] = calculate_macd(close)
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = calculate_bollinger(close)
    return df

@st.cache_data(ttl=3600)
def fetch_sentiment():
    try:
        s = get_daily_sentiment()
        return float(s['Sentiment_Score'].iloc[-1])
    except:
        return 0.0

def live_binance(symbol="BTCUSDT"):
    try:
        r = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}", timeout=2)
        d = r.json()
        return float(d['lastPrice']), float(d['priceChangePercent']), float(d['volume'])
    except:
        return None, None, None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=52)
    st.markdown(
        '<div style="font-family:Space Grotesk,sans-serif;font-weight:800;'
        'font-size:1.1rem;color:#e6edf3;margin:4px 0 16px;">⚡ Trading Terminal</div>',
        unsafe_allow_html=True
    )

    st.markdown("#### ⚙️ Configuration")
    TICKER_MAP = {
        "Bitcoin (BTC)": ("BTC-USD", "BTCUSDT"),
        "Ethereum (ETH)": ("ETH-USD", "ETHUSDT"),
        "Solana (SOL)": ("SOL-USD", "SOLUSDT"),
        "BNB (BNB)": ("BNB-USD", "BNBUSDT"),
    }
    selected_label = st.selectbox("Asset", list(TICKER_MAP.keys()))
    yf_ticker, binance_symbol = TICKER_MAP[selected_label]
    lookback = st.slider("Lookback (days)", 30, 90, 60, 5)
    st.markdown("---")

    # Load data
    model       = load_model()
    market_data = fetch_data(yf_ticker, lookback)
    sentiment   = fetch_sentiment()

    st.markdown("####  System Status")
    for name, ok in [
        ("XGBoost Brain Online",  model is not None),
        ("YFinance Data Synced",  not market_data.empty),
        ("NLP Scraper Active",    sentiment != 0.0),
    ]:
        cls = "sys-ok" if ok else "sys-fail"
        ico = "" if ok else ""
        st.markdown(f'<div class="{cls}">{ico} {name}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick stats
    rsi_now  = float(market_data['RSI'].squeeze().iloc[-1])
    macd_now = float(market_data['MACD'].squeeze().iloc[-1])
    sig_now  = float(market_data['MACD_Sig'].squeeze().iloc[-1])
    mdd      = max_drawdown(market_data['Close'].squeeze())
    cumret   = float(market_data['Cum_Return'].squeeze().iloc[-1]) * 100
    vol_7d   = float(market_data['Volatility_7d'].squeeze().iloc[-1]) * 100

    st.markdown("####  Quick Stats")
    stats = [
        ("RSI (14)",       f"{rsi_now:.1f}",    "Overbought" if rsi_now > 70 else " Oversold" if rsi_now < 30 else "Neutral"),
        ("MACD",           f"{macd_now:.1f}",   "▲ Bull" if macd_now > sig_now else "▼ Bear"),
        ("Max Drawdown",   f"{mdd:.2f}%",        ""),
        ("Period Return",  f"{cumret:+.2f}%",   ""),
        ("7d Volatility",  f"{vol_7d:.2f}%",    ""),
    ]
    for label, val, hint in stats:
        color = "#22c55e" if "+" in val or "Bull" in hint or "Oversold" in hint else \
                "#ef4444" if "-" in val or "Bear" in hint or "Overbought" in hint else "#e6edf3"
        hint_html = f'<span style="color:#4e5d6c;font-size:0.72rem;"> {hint}</span>' if hint else ""
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid #1c2330;">'
            f'<span style="color:#4e5d6c;font-size:0.8rem;font-family:JetBrains Mono,monospace;">{label}</span>'
            f'<span style="color:{color};font-weight:700;font-size:0.82rem;'
            f'font-family:JetBrains Mono,monospace;">{val}{hint_html}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    auto_refresh = st.toggle(" Auto-Refresh (5 min)", value=False)
    if auto_refresh:
        time.sleep(300)
        st.rerun()

    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")


# ---------------------------------------------------------
# LIVE PRICE
# ---------------------------------------------------------
if market_data.empty:
    st.error(" Market data unavailable. Check connection."); st.stop()

live_price, chg_pct, bin_vol = live_binance(binance_symbol)
if live_price is None:
    live_price = float(market_data['Close'].squeeze().iloc[-1])
    chg_pct    = float(market_data['Daily_Return'].squeeze().iloc[-1]) * 100
    bin_vol    = float(market_data['Volume'].squeeze().iloc[-1])

fg_val, fg_label = get_fear_greed()

# Derived scalars
close_s    = market_data['Close'].squeeze()
daily_ret  = market_data['Daily_Return'].squeeze()
sma7_last  = float(market_data['SMA_7'].squeeze().iloc[-1])
sma14_last = float(market_data['SMA_14'].squeeze().iloc[-1])
bb_up      = float(market_data['BB_Upper'].squeeze().iloc[-1])
bb_lo      = float(market_data['BB_Lower'].squeeze().iloc[-1])
bb_mid     = float(market_data['BB_Mid'].squeeze().iloc[-1])
bb_width   = ((bb_up - bb_lo) / bb_mid) * 100
sig_last   = float(market_data['MACD_Sig'].squeeze().iloc[-1])
p_high     = float(market_data['High'].squeeze().max())
p_low      = float(market_data['Low'].squeeze().min())

# Market regime
bull_count = sum([sma7_last > sma14_last, rsi_now > 50, macd_now > sig_last, live_price > sma14_last])
regime = ("BULLISH", "#22c55e") if bull_count >= 3 else \
         ("BEARISH", "#ef4444") if bull_count <= 1 else \
         ("NEUTRAL", "#eab308")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("#  AI Algorithmic Trading Terminal")

fg_str = f"Fear & Greed: {fg_val} — {fg_label}" if fg_val else "Fear & Greed: N/A"
st.markdown(
    f'<div style="background:#0d1117;border:1px solid #1c2330;border-radius:10px;'
    f'padding:8px 18px;font-family:JetBrains Mono,monospace;font-size:0.78rem;'
    f'color:#4e5d6c;margin-bottom:1.2rem;display:flex;gap:24px;flex-wrap:wrap;">'
    f'<span> {datetime.now().strftime("%d %b %Y · %H:%M:%S")}</span>'
    f'<span> {selected_label}</span>'
    f'<span>Regime: <b style="color:{regime[1]};">{regime[0]}</b></span>'
    f'<span> {fg_str}</span>'
    f'</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# KPI ROW 1
# ---------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(" Live Price",      f"${live_price:,.2f}",  f"{chg_pct:+.2f}%")
c2.metric(" 24h Volume",      f"${bin_vol/1e9:.2f}B" if bin_vol and bin_vol > 1e9 else f"${bin_vol:,.0f}" if bin_vol else "N/A")
c3.metric(" 7d Volatility",  f"{vol_7d:.2f}%",
          " High" if vol_7d > 4 else (" Moderate" if vol_7d > 2 else " Low"))
c4.metric(" NLP Sentiment",  f"{sentiment:+.3f}",
          "Positive" if sentiment > 0.05 else ("Negative" if sentiment < -0.05 else "Neutral"))
c5.metric(" RSI (14)",        f"{rsi_now:.1f}",
          "Overbought" if rsi_now > 70 else ("Oversold" if rsi_now < 30 else "Neutral"))

st.markdown("<br>", unsafe_allow_html=True)

# KPI ROW 2
c6, c7, c8, c9, c10 = st.columns(5)
c6.metric(f" {lookback}d High",  f"${p_high:,.2f}")
c7.metric(f" {lookback}d Low",   f"${p_low:,.2f}")
c8.metric(" BB Width",           f"{bb_width:.2f}%",
          "Expanding" if bb_width > 5 else "Compressing")
c9.metric(" Max Drawdown",        f"{mdd:.2f}%")
c10.metric(" Fear & Greed",       str(fg_val) if fg_val else "N/A",
           fg_label if fg_label != "N/A" else "Unavailable")

st.divider()


# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------
st.subheader(" Market Technicals")

tab1, tab2, tab3 = st.tabs(["🕯️  Price · Volume · BBands", "📉  RSI", "〰️  MACD"])

# ── Tab 1: Price + Bollinger + Volume ──────────────────────────────────────
with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.025,
                        subplot_titles=("Price Action", "Volume"))

    # Bollinger fill
    fig.add_trace(go.Scatter(
        x=market_data.index, y=market_data['BB_Upper'].squeeze(),
        line=dict(color='rgba(99,102,241,0)', width=0), showlegend=False, name='BB Upper'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=market_data.index, y=market_data['BB_Lower'].squeeze(),
        line=dict(color='rgba(99,102,241,0)', width=0),
        fill='tonexty', fillcolor='rgba(99,102,241,0.06)',
        name='Bollinger Band', showlegend=True
    ), row=1, col=1)

    # BB borders
    for col_name, dash in [('BB_Upper', 'dot'), ('BB_Lower', 'dot'), ('BB_Mid', 'dash')]:
        fig.add_trace(go.Scatter(
            x=market_data.index, y=market_data[col_name].squeeze(),
            line=dict(color='rgba(99,102,241,0.5)', width=1, dash=dash),
            showlegend=False, name=col_name
        ), row=1, col=1)

    # Candles
    fig.add_trace(go.Candlestick(
        x=market_data.index,
        open=market_data['Open'].squeeze(), high=market_data['High'].squeeze(),
        low=market_data['Low'].squeeze(),  close=close_s,
        increasing=dict(line=dict(color='#22c55e', width=1.2), fillcolor='#22c55e'),
        decreasing=dict(line=dict(color='#ef4444', width=1.2), fillcolor='#ef4444'),
        name="Price"
    ), row=1, col=1)

    # SMAs
    for col_name, color, nm in [('SMA_7','#f7931a','SMA 7'), ('SMA_14','#06b6d4','SMA 14')]:
        fig.add_trace(go.Scatter(
            x=market_data.index, y=market_data[col_name].squeeze(),
            line=dict(color=color, width=1.5), name=nm
        ), row=1, col=1)

    # Volume
    bar_colors = ['#22c55e' if r >= 0 else '#ef4444' for r in daily_ret]
    fig.add_trace(go.Bar(
        x=market_data.index, y=market_data['Volume'].squeeze(),
        marker_color=bar_colors, opacity=0.65, name='Volume'
    ), row=2, col=1)

    fig.update_layout(
        height=580, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.03, x=0, bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        **PLOTLY_DARK
    )
    apply_grid(fig)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: RSI ─────────────────────────────────────────────────────────────
with tab2:
    rsi_s = market_data['RSI'].squeeze()
    fig_rsi = go.Figure()

    fig_rsi.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.07)', line_width=0)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor='rgba(34,197,94,0.07)',  line_width=0)

    fig_rsi.add_trace(go.Scatter(
        x=market_data.index, y=rsi_s,
        line=dict(color='#a78bfa', width=2.2),
        fill='tozeroy', fillcolor='rgba(167,139,250,0.06)', name='RSI 14'
    ))

    for val, color, text in [(70, 'rgba(239,68,68,0.55)',   'Overbought (70)'),
                              (30, 'rgba(34,197,94,0.55)',   'Oversold (30)'),
                              (50, 'rgba(78,93,108,0.55)',   '')]:
        fig_rsi.add_hline(y=val, line=dict(color=color, dash='dash', width=1),
                          annotation_text=text, annotation_font_size=10)

    fig_rsi.update_layout(
        height=380, xaxis_rangeslider_visible=False, **PLOTLY_DARK
    )
    apply_grid(fig_rsi)
    fig_rsi.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.caption(f"Current RSI: **{rsi_now:.2f}** — {'Overbought ' if rsi_now > 70 else 'Oversold ' if rsi_now < 30 else 'Neutral zone'}")

# ── Tab 3: MACD ────────────────────────────────────────────────────────────
with tab3:
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.55, 0.45], vertical_spacing=0.04,
                             subplot_titles=("MACD Line vs Signal", "Histogram"))

    fig_macd.add_trace(go.Scatter(
        x=market_data.index, y=market_data['MACD'].squeeze(),
        line=dict(color='#06b6d4', width=2), name='MACD'
    ), row=1, col=1)
    fig_macd.add_trace(go.Scatter(
        x=market_data.index, y=market_data['MACD_Sig'].squeeze(),
        line=dict(color='#f7931a', width=1.5, dash='dot'), name='Signal'
    ), row=1, col=1)

    hist_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in market_data['MACD_Hist'].squeeze()]
    fig_macd.add_trace(go.Bar(
        x=market_data.index, y=market_data['MACD_Hist'].squeeze(),
        marker_color=hist_colors, opacity=0.75, name='Histogram'
    ), row=2, col=1)

    fig_macd.update_layout(height=420, xaxis_rangeslider_visible=False,
                           legend=dict(orientation='h', y=1.05), **PLOTLY_DARK)
    apply_grid(fig_macd)
    st.plotly_chart(fig_macd, use_container_width=True)


st.divider()


# ---------------------------------------------------------
# TECHNICAL SIGNAL DASHBOARD
# ---------------------------------------------------------
st.subheader(" Technical Signal Dashboard")

sc1, sc2, sc3, sc4 = st.columns(4)

# Compute signals
sma_bull  = sma7_last > sma14_last
rsi_bull  = rsi_now < 30;  rsi_bear = rsi_now > 70
macd_bull = macd_now > sig_last
bb_bull   = live_price < bb_lo;  bb_bear = live_price > bb_up

signals = [
    (sc1, "SMA Crossover",
     signal_html("Golden Cross" if sma_bull else "Death Cross", sma_bull, not sma_bull),
     f"SMA7 ${sma7_last:,.0f} · SMA14 ${sma14_last:,.0f}"),
    (sc2, "RSI Zone",
     signal_html("Oversold — Buy" if rsi_bull else ("Overbought — Sell" if rsi_bear else f"Neutral {rsi_now:.1f}"),
                 rsi_bull, rsi_bear),
     f"RSI: {rsi_now:.2f}"),
    (sc3, "MACD",
     signal_html("Bullish Cross" if macd_bull else "Bearish Cross", macd_bull, not macd_bull),
     f"MACD {macd_now:.1f} · Sig {sig_last:.1f}"),
    (sc4, "Bollinger Position",
     signal_html("Below Lower — Bounce" if bb_bull else ("Above Upper — Caution" if bb_bear else "Within Bands"),
                 bb_bull, bb_bear),
     f"Upper ${bb_up:,.0f} · Lower ${bb_lo:,.0f}"),
]

for col, title, badge, caption in signals:
    with col:
        st.markdown(
            f'<div class="card">'
            f'<div style="color:#4e5d6c;font-size:0.72rem;font-family:JetBrains Mono,monospace;'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">{title}</div>'
            f'{badge}'
            f'<div style="color:#4e5d6c;font-size:0.73rem;font-family:JetBrains Mono,monospace;margin-top:8px;">{caption}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# Signal Consensus Bar
total_bull_sigs = sum([sma_bull, rsi_bull or rsi_now > 50, macd_bull, not bb_bear])
consensus_color = "#22c55e" if total_bull_sigs >= 3 else "#ef4444" if total_bull_sigs <= 1 else "#eab308"
consensus_pct   = total_bull_sigs / 4 * 100
consensus_label = "STRONG BULL" if total_bull_sigs >= 3 else ("STRONG BEAR" if total_bull_sigs <= 1 else "MIXED")

st.markdown(
    f'<div class="card" style="margin-top:0.5rem;">'
    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
    f'<span style="color:#4e5d6c;font-family:JetBrains Mono,monospace;font-size:0.78rem;text-transform:uppercase;">Signal Consensus</span>'
    f'<span style="color:{consensus_color};font-weight:800;font-family:JetBrains Mono,monospace;">{total_bull_sigs}/4 Bullish — {consensus_label}</span>'
    f'</div>'
    f'<div style="background:#1c2330;border-radius:6px;height:10px;overflow:hidden;">'
    f'<div style="background:linear-gradient(90deg,{consensus_color}88,{consensus_color});'
    f'width:{consensus_pct:.0f}%;height:100%;border-radius:6px;"></div>'
    f'</div></div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------------
# AI PREDICTION ENGINE
# ---------------------------------------------------------
st.subheader(" AI Predictive Engine")

pred_col, info_col = st.columns([3, 2])

with pred_col:
    if st.button(" Execute XGBoost Inference", type="primary", use_container_width=True):
        with st.spinner(" Compiling feature tensor · Running XGBoost inference…"):

            live_features = prepare_live_features(
                current_price    = live_price,
                current_volume   = float(market_data['Volume'].squeeze().iloc[-1]),
                sma_7            = sma7_last,
                sma_14           = sma14_last,
                daily_return     = float(daily_ret.iloc[-1]),
                yesterday_sentiment = sentiment,
                volatility_7d    = float(market_data['Volatility_7d'].squeeze().iloc[-1]),
                return_lag1      = float(daily_ret.iloc[-2])
            )

            prediction  = model.predict(live_features)[0]
            probability = model.predict_proba(live_features)[0]
            conf        = float(probability[1]) if prediction == 1 else float(probability[0])

        bull = prediction == 1
        verdict_color = "#22c55e" if bull else "#ef4444"
        verdict_bg    = "rgba(34,197,94,0.07)"  if bull else "rgba(239,68,68,0.07)"
        verdict_border = "rgba(34,197,94,0.3)"  if bull else "rgba(239,68,68,0.3)"
        verdict_text  = " BULLISH — Price Expected to Rise" if bull else " BEARISH — Price Expected to Fall"

        st.markdown(
            f'<div style="background:{verdict_bg};border:1px solid {verdict_border};'
            f'border-radius:14px;padding:1.4rem 1.6rem;margin:1rem 0;">'
            f'<div style="font-family:Space Grotesk,sans-serif;font-weight:800;'
            f'font-size:1.35rem;color:{verdict_color};">{verdict_text}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Confidence bar
        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:10px;">'
            f'<span style="color:#4e5d6c;font-family:JetBrains Mono,monospace;font-size:0.78rem;">MODEL CONFIDENCE</span>'
            f'<span style="color:{verdict_color};font-weight:800;font-family:JetBrains Mono,monospace;font-size:1.1rem;">{conf*100:.1f}%</span>'
            f'</div>'
            f'<div style="background:#1c2330;border-radius:6px;height:14px;overflow:hidden;">'
            f'<div style="background:linear-gradient(90deg,{verdict_color}77,{verdict_color});'
            f'width:{conf*100:.0f}%;height:100%;border-radius:6px;"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:8px;">'
            f'<span style="color:#4e5d6c;font-size:0.7rem;font-family:JetBrains Mono,monospace;">0%</span>'
            f'<span style="color:#4e5d6c;font-size:0.7rem;font-family:JetBrains Mono,monospace;">50%</span>'
            f'<span style="color:#4e5d6c;font-size:0.7rem;font-family:JetBrains Mono,monospace;">100%</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        # Probability breakdown
        pc1, pc2 = st.columns(2)
        pc1.metric("P(Bullish)", f"{probability[1]*100:.1f}%")
        pc2.metric("P(Bearish)", f"{probability[0]*100:.1f}%")

        with st.expander(" Feature Matrix (Live Tensor)"):
            st.markdown('<p style="color:#4e5d6c;font-size:0.8rem;font-family:JetBrains Mono,monospace;">Normalized input vector passed to XGBoost:</p>', unsafe_allow_html=True)
            st.dataframe(live_features, use_container_width=True)

with info_col:
    # Model card
    st.markdown(
        '<div class="card">'
        '<div style="font-family:Space Grotesk,sans-serif;font-weight:700;color:#e6edf3;margin-bottom:12px;"> Model Card</div>'
        '<table style="width:100%;border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.78rem;">'
        + "".join([
            f'<tr><td style="color:#4e5d6c;padding:5px 0;">{k}</td>'
            f'<td style="color:#e6edf3;text-align:right;">{v}</td></tr>'
            for k, v in [
                ("Algorithm",    "XGBoost Classifier"),
                ("Features",     "8-dim tensor"),
                ("Target",       "Next-day direction"),
                ("NLP Feed",     "News sentiment"),
                ("Price Feed",   "Binance · YFinance"),
                ("Inference",    "On-demand"),
            ]
        ])
        + '</table></div>',
        unsafe_allow_html=True
    )

    # Signal consensus box
    st.markdown(
        f'<div class="card" style="text-align:center;margin-top:0.5rem;">'
        f'<div style="font-family:Space Grotesk,sans-serif;font-weight:700;color:#e6edf3;margin-bottom:12px;">🎯 Signal Consensus</div>'
        f'<div style="font-size:3rem;font-weight:800;font-family:JetBrains Mono,monospace;color:{consensus_color};">'
        f'{total_bull_sigs}/4</div>'
        f'<div style="color:#4e5d6c;font-size:0.8rem;margin:4px 0;">Bullish Technical Signals</div>'
        f'<div style="color:{consensus_color};font-weight:700;font-size:0.9rem;margin-top:6px;">{consensus_label}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="background:rgba(234,179,8,0.08);border:1px solid rgba(234,179,8,0.25);'
        'border-radius:10px;padding:12px;margin-top:8px;">'
        '<p style="color:#eab308;font-size:0.75rem;font-family:JetBrains Mono,monospace;margin:0;">'
        ' For educational purposes only. This is not financial advice.</p></div>',
        unsafe_allow_html=True
    )

st.divider()


# ---------------------------------------------------------
# RETURNS ANALYSIS
# ---------------------------------------------------------
st.subheader(" Returns & Distribution Analysis")

ra1, ra2 = st.columns(2)

with ra1:
    cum_pct = market_data['Cum_Return'].squeeze() * 100
    fig_cum = go.Figure(go.Scatter(
        x=market_data.index, y=cum_pct,
        fill='tozeroy',
        fillcolor='rgba(247,147,26,0.08)',
        line=dict(color='#f7931a', width=2.2),
        name='Cumulative Return'
    ))
    fig_cum.add_hline(y=0, line=dict(color='#4e5d6c', dash='dot', width=1))
    fig_cum.update_layout(
        title="Cumulative Return (%)", height=320, **PLOTLY_DARK
    )
    apply_grid(fig_cum)
    fig_cum.update_yaxes(ticksuffix='%')
    st.plotly_chart(fig_cum, use_container_width=True)

with ra2:
    dr_pct = daily_ret.dropna() * 100
    fig_hist = go.Figure(go.Histogram(
        x=dr_pct, nbinsx=30,
        marker_color='#6366f1', opacity=0.82,
        name='Daily Returns'
    ))
    fig_hist.add_vline(x=float(dr_pct.mean()),
                       line=dict(color='#f7931a', dash='dash'),
                       annotation_text=f"μ={dr_pct.mean():.2f}%",
                       annotation_font_color='#f7931a', annotation_font_size=11)
    fig_hist.update_layout(
        title="Daily Returns Distribution", height=320, **PLOTLY_DARK
    )
    apply_grid(fig_hist)
    fig_hist.update_xaxes(ticksuffix='%')
    st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown(
    '<div style="text-align:center;color:#2a3340;font-family:JetBrains Mono,monospace;'
    'font-size:0.72rem;padding:1rem 0;letter-spacing:0.05em;">'
    ' AI ALGORITHMIC TRADING TERMINAL &nbsp;·&nbsp; '
    'XGBOOST + NLP SENTIMENT ENGINE &nbsp;·&nbsp; '
    ' FOR EDUCATIONAL PURPOSES ONLY'
    '</div>',
    unsafe_allow_html=True
)