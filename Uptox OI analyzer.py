"""
╔══════════════════════════════════════════════════════════════╗
║         UPSTOX OI ANALYZER — Live Options Intelligence      ║
║  Pulls live OI data, analyzes CE/PE buildup around ATM,     ║
║  and scores directional bias using a 4-factor model.        ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import json

# ── Page Config ──
st.set_page_config(
    page_title="Upstox OI Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;600;700&display=swap');

/* Global */
.stApp { background: #0a0e17; }
section[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1e293b; }
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: #111827; border: 1px solid #1e293b; border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="stMetric"] label { color: #64748b !important; font-size: 11px !important; letter-spacing: 1.5px; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important;
    color: #e2e8f0 !important;
}

/* Tables */
table { border-collapse: collapse !important; width: 100% !important; }
thead tr th {
    background: #111827 !important; color: #64748b !important;
    font-size: 11px !important; letter-spacing: 1px; padding: 10px 8px !important;
    border-bottom: 1px solid #1e293b !important; text-align: right !important;
}
tbody tr td {
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
    padding: 8px !important; border-bottom: 1px solid #1e293b33 !important;
    color: #e2e8f0 !important; text-align: right !important;
}
tbody tr:hover { background: #38bdf808 !important; }

/* Direction card */
.direction-card {
    border-radius: 14px; padding: 24px; margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
}
.score-label { font-size: 11px; letter-spacing: 2px; color: #64748b; margin-bottom: 4px; }
.score-value { font-size: 40px; font-weight: 700; }
.direction-text { font-size: 28px; font-weight: 700; }
.sentiment-text { font-size: 13px; color: #94a3b8; margin-top: 4px; }

/* Sidebar */
.sidebar-header { font-size: 12px; letter-spacing: 3px; color: #38bdf8; font-weight: 600; }
.sidebar-title { font-size: 22px; font-weight: 700; color: #e2e8f0; margin-top: 4px; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Buildup bar */
.buildup-bar {
    height: 10px; border-radius: 5px; overflow: hidden; display: flex; margin: 8px 0;
}
.buildup-ce { background: #f43f5e; height: 100%; }
.buildup-pe { background: #22c55e; height: 100%; }

/* Info box */
.info-box {
    background: #111827; border: 1px solid #1e293b; border-radius: 12px;
    padding: 16px 20px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Index Definitions ──
INDICES = {
    "NIFTY 50": {"key": "NSE_INDEX|Nifty 50", "symbol": "NIFTY", "diff": 50},
    "BANK NIFTY": {"key": "NSE_INDEX|Nifty Bank", "symbol": "BANKNIFTY", "diff": 100},
    "FINNIFTY": {"key": "NSE_INDEX|Nifty Fin Service", "symbol": "FINNIFTY", "diff": 50},
    "MIDCAP NIFTY": {"key": "NSE_INDEX|NIFTY MID SELECT", "symbol": "MIDCPNIFTY", "diff": 25},
}

# ── Upstox API Helper ──
class UpstoxClient:
    BASE = "https://api.upstox.com/v2"

    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def get_spot_price(self, instrument_key: str):
        url = f"{self.BASE}/market-quote/quotes"
        r = requests.get(url, headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        quote_key = list(data.get("data", {}).keys())[0]
        return data["data"][quote_key]["last_price"]

    def get_expiries(self, instrument_key: str):
        url = f"{self.BASE}/option/contract"
        r = requests.get(url, headers=self.headers, params={"instrument_key": instrument_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
        expiries = sorted(set(
            c.get("expiry", "")[:10] if isinstance(c.get("expiry"), str) else str(c.get("expiry", ""))[:10]
            for c in data.get("data", [])
        ))
        return [e for e in expiries if e and e != "None"]

    def get_option_chain(self, instrument_key: str, expiry_date: str):
        url = f"{self.BASE}/option/chain"
        params = {"instrument_key": instrument_key, "expiry_date": expiry_date}
        r = requests.get(url, headers=self.headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])


# ── Analysis Engine ──
def analyze_oi(strikes_df: pd.DataFrame, atm: float, depth: int = 5):
    """
    4-Factor Composite Directional Scoring Model
    ─────────────────────────────────────────────
    1. PCR (30%)          — Put OI / Call OI ratio around ATM
    2. Max OI Walls (25%) — Distance of max CE/PE OI walls from ATM
    3. OI Buildup (25%)   — Fresh OI addition on CE vs PE side
    4. Volume PCR (20%)   — Active trading volume confirmation

    Score: -100 (strong bearish) to +100 (strong bullish)
    """
    if strikes_df.empty:
        return None

    atm_idx = (strikes_df["strike"] - atm).abs().idxmin()
    atm_pos = strikes_df.index.get_loc(atm_idx)
    start = max(0, atm_pos - depth)
    end = min(len(strikes_df), atm_pos + depth + 1)
    window = strikes_df.iloc[start:end].copy()

    total_ce_oi = window["ce_oi"].sum()
    total_pe_oi = window["pe_oi"].sum()
    total_ce_vol = window["ce_volume"].sum()
    total_pe_vol = window["pe_volume"].sum()

    max_ce_row = window.loc[window["ce_oi"].idxmax()]
    max_pe_row = window.loc[window["pe_oi"].idxmax()]

    ce_buildup = window.loc[window["ce_oi_chg"] > 0, "ce_oi_chg"].sum()
    pe_buildup = window.loc[window["pe_oi_chg"] > 0, "pe_oi_chg"].sum()

    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    vol_pcr = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0

    # ── Scoring ──
    score = 0

    # 1. PCR (30%)
    if pcr > 1.3: score += 30
    elif pcr > 1.1: score += 15
    elif pcr < 0.7: score -= 30
    elif pcr < 0.9: score -= 15

    # 2. Max OI Walls (25%)
    resistance_dist = max_ce_row["strike"] - atm
    support_dist = atm - max_pe_row["strike"]
    if resistance_dist > support_dist * 1.5: score += 25
    elif support_dist > resistance_dist * 1.5: score -= 25
    elif resistance_dist > support_dist: score += 10
    else: score -= 10

    # 3. OI Buildup (25%)
    total_buildup = ce_buildup + pe_buildup
    if total_buildup > 0:
        pe_ratio = pe_buildup / total_buildup
        if pe_ratio > 0.65: score += 25
        elif pe_ratio > 0.55: score += 12
        elif pe_ratio < 0.35: score -= 25
        elif pe_ratio < 0.45: score -= 12

    # 4. Volume PCR (20%)
    if vol_pcr > 1.2: score += 20
    elif vol_pcr > 1.0: score += 10
    elif vol_pcr < 0.8: score -= 20
    elif vol_pcr < 1.0: score -= 10

    # Direction mapping
    if score >= 40: direction, sentiment, color = "STRONG BULLISH", "Bulls firmly in control — heavy put writing as support", "#00e676"
    elif score >= 20: direction, sentiment, color = "BULLISH", "Bullish bias — put OI buildup confirms support", "#66ffa6"
    elif score >= 5: direction, sentiment, color = "MILDLY BULLISH", "Slight bullish tilt — watch for confirmation", "#b9f6ca"
    elif score <= -40: direction, sentiment, color = "STRONG BEARISH", "Bears firmly in control — heavy call writing as resistance", "#ff1744"
    elif score <= -20: direction, sentiment, color = "BEARISH", "Bearish bias — call OI buildup confirms resistance", "#ff6e7a"
    elif score <= -5: direction, sentiment, color = "MILDLY BEARISH", "Slight bearish tilt — watch for confirmation", "#ffcdd2"
    else: direction, sentiment, color = "NEUTRAL", "No clear directional bias — market is range-bound", "#ffd740"

    return {
        "pcr": round(pcr, 3),
        "vol_pcr": round(vol_pcr, 3),
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "max_ce_oi": int(max_ce_row["ce_oi"]),
        "max_pe_oi": int(max_pe_row["pe_oi"]),
        "max_ce_strike": int(max_ce_row["strike"]),
        "max_pe_strike": int(max_pe_row["strike"]),
        "ce_buildup": ce_buildup,
        "pe_buildup": pe_buildup,
        "score": score,
        "direction": direction,
        "sentiment": sentiment,
        "color": color,
        "window": window,
    }


def fmt_lakh(n):
    if pd.isna(n) or n is None: return "—"
    v = float(n)
    if abs(v) >= 100000: return f"{v/100000:.2f}L"
    if abs(v) >= 1000: return f"{v/1000:.1f}K"
    return f"{v:,.0f}"


# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="sidebar-header">UPSTOX</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">OI Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")

    token = st.text_input(
        "🔑 Access Token",
        type="password",
        placeholder="Paste Upstox access_token...",
        help="Generate daily from Upstox Developer Console → Login → Get access_token"
    )

    if token:
        st.success("Token set ✓", icon="🔒")

    st.markdown("---")

    selected_index = st.selectbox("📈 Select Index", list(INDICES.keys()), index=0)
    idx = INDICES[selected_index]

    analysis_depth = st.select_slider(
        "🎯 Analysis Depth (strikes around ATM)",
        options=[3, 5, 7, 10],
        value=5,
        help="Number of strikes on each side of ATM to include in directional analysis"
    )

    auto_refresh = st.toggle("⏱️ Auto-refresh (30s)", value=False)
    refresh_btn = st.button("🔄 Refresh Now", width="stretch", type="primary")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#64748b; line-height:1.7;">
    <b style="color:#38bdf8;">Scoring Model</b><br>
    • PCR (30%) — Put/Call OI ratio<br>
    • Max OI Walls (25%) — Support & Resistance<br>
    • OI Buildup (25%) — Smart money flow<br>
    • Volume PCR (20%) — Trading confirmation<br>
    <br>Score: −100 → +100
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ──
if not token:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <div style="font-size:14px; letter-spacing:4px; color:#38bdf8; font-weight:600;">UPSTOX</div>
        <h1 style="color:#e2e8f0; font-size:36px; margin:8px 0 16px;">OI Analyzer</h1>
        <p style="color:#64748b; font-size:15px; max-width:500px; margin:0 auto; line-height:1.7;">
            Enter your Upstox API access token in the sidebar to begin live
            Open Interest analysis with directional scoring.
        </p>
        <div style="margin-top:40px; background:#111827; border:1px solid #1e293b; border-radius:14px;
                    padding:24px; max-width:500px; margin-left:auto; margin-right:auto; text-align:left;">
            <p style="color:#94a3b8; font-size:13px; line-height:1.8;">
            <b style="color:#38bdf8;">How to get your token:</b><br>
            1. Go to <code>Upstox Developer Console</code><br>
            2. Create an app & get API Key + Secret<br>
            3. Complete the OAuth login flow<br>
            4. Copy the <code>access_token</code> from the response<br>
            5. Paste it in the sidebar
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Fetch Data ──
client = UpstoxClient(token)

@st.cache_data(ttl=25, show_spinner=False)
def fetch_all_data(_client, instrument_key, expiry, _ts):
    """Fetches spot price, expiries, and option chain. _ts forces cache refresh."""
    spot = _client.get_spot_price(instrument_key)

    if not expiry:
        expiries = _client.get_expiries(instrument_key)
        expiry = expiries[0] if expiries else ""
    else:
        expiries = _client.get_expiries(instrument_key)

    chain = _client.get_option_chain(instrument_key, expiry)
    return spot, expiries, expiry, chain


try:
    # Get stored expiry or blank
    exp_key = f"expiry_{selected_index}"
    stored_expiry = st.session_state.get(exp_key, "")

    ts = int(time.time() // 30) if auto_refresh else "manual"
    if refresh_btn:
        ts = time.time()

    with st.spinner("Fetching live data from Upstox..."):
        spot, expiries, used_expiry, chain_data = fetch_all_data(
            client, idx["key"], stored_expiry, ts
        )

    # Expiry selector (now that we have the list)
    if expiries:
        sel_exp_idx = expiries.index(used_expiry) if used_expiry in expiries else 0
        with st.sidebar:
            chosen_expiry = st.selectbox("📅 Expiry", expiries, index=sel_exp_idx)
            if chosen_expiry != stored_expiry:
                st.session_state[exp_key] = chosen_expiry
                st.rerun()

    # Parse chain into DataFrame
    rows = []
    for item in chain_data:
        sp = item.get("strike_price", 0)
        row = {"strike": sp}

        cd = item.get("call_options", {}).get("market_data", {})
        cg = item.get("call_options", {}).get("option_greeks", {})
        row["ce_oi"] = cd.get("oi", 0) or 0
        row["ce_oi_chg"] = cd.get("oi_change", cd.get("oi_day_change", 0)) or 0
        row["ce_volume"] = cd.get("volume", 0) or 0
        row["ce_ltp"] = cd.get("ltp", 0) or 0
        row["ce_iv"] = cg.get("iv", 0) or 0

        pd_data = item.get("put_options", {}).get("market_data", {})
        pg = item.get("put_options", {}).get("option_greeks", {})
        row["pe_oi"] = pd_data.get("oi", 0) or 0
        row["pe_oi_chg"] = pd_data.get("oi_change", pd_data.get("oi_day_change", 0)) or 0
        row["pe_volume"] = pd_data.get("volume", 0) or 0
        row["pe_ltp"] = pd_data.get("ltp", 0) or 0
        row["pe_iv"] = pg.get("iv", 0) or 0

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    atm = round(spot / idx["diff"]) * idx["diff"]

    # Trim to ~20 strikes around ATM
    atm_idx = (df["strike"] - atm).abs().idxmin()
    atm_pos = df.index.get_loc(atm_idx)
    trim_start = max(0, atm_pos - 10)
    trim_end = min(len(df), atm_pos + 11)
    display_df = df.iloc[trim_start:trim_end].copy()

    # ── Top Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{selected_index} SPOT", f"₹{spot:,.2f}")
    c2.metric("ATM STRIKE", f"{atm:,}")
    c3.metric("EXPIRY", used_expiry)
    c4.metric("UPDATED", datetime.now().strftime("%H:%M:%S"))

    # ── Analysis ──
    result = analyze_oi(df, atm, analysis_depth)

    if result:
        st.markdown("---")

        # Direction Signal
        bg_alpha = "22"
        border_color = result["color"]
        st.markdown(f"""
        <div class="direction-card" style="background: linear-gradient(135deg, #111827, {result['color']}11);
             border: 1px solid {result['color']}44;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <div class="score-label">DIRECTIONAL SIGNAL</div>
                    <div class="direction-text" style="color:{result['color']};">{result['direction']}</div>
                    <div class="sentiment-text">{result['sentiment']}</div>
                </div>
                <div style="text-align:right;">
                    <div class="score-label">COMPOSITE SCORE</div>
                    <div class="score-value" style="color:{result['color']};">
                        {"+" if result['score'] > 0 else ""}{result['score']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Analysis metrics
        m1, m2, m3, m4 = st.columns(4)

        pcr_color = "#22c55e" if result["pcr"] > 1 else "#f43f5e"
        m1.markdown(f"""
        <div class="info-box">
            <div style="font-size:11px; color:#64748b; letter-spacing:1.5px;">PCR (OI)</div>
            <div style="font-size:28px; font-weight:700; font-family:'JetBrains Mono'; color:{pcr_color};">{result['pcr']}</div>
            <div style="font-size:11px; color:#64748b;">{'Put writing dominant → Bullish' if result['pcr']>1 else 'Call writing dominant → Bearish'}</div>
        </div>
        """, unsafe_allow_html=True)

        vpc_color = "#22c55e" if result["vol_pcr"] > 1 else "#f43f5e"
        m2.markdown(f"""
        <div class="info-box">
            <div style="font-size:11px; color:#64748b; letter-spacing:1.5px;">VOLUME PCR</div>
            <div style="font-size:28px; font-weight:700; font-family:'JetBrains Mono'; color:{vpc_color};">{result['vol_pcr']}</div>
            <div style="font-size:11px; color:#64748b;">Active trading confirmation</div>
        </div>
        """, unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="info-box">
            <div style="font-size:11px; color:#f43f5e; letter-spacing:1.5px;">RESISTANCE (Max CE OI)</div>
            <div style="font-size:22px; font-weight:700; font-family:'JetBrains Mono'; color:#e2e8f0;">{result['max_ce_strike']:,}</div>
            <div style="font-size:11px; color:#64748b;">{fmt_lakh(result['max_ce_oi'])} contracts</div>
        </div>
        """, unsafe_allow_html=True)

        m4.markdown(f"""
        <div class="info-box">
            <div style="font-size:11px; color:#22c55e; letter-spacing:1.5px;">SUPPORT (Max PE OI)</div>
            <div style="font-size:22px; font-weight:700; font-family:'JetBrains Mono'; color:#e2e8f0;">{result['max_pe_strike']:,}</div>
            <div style="font-size:11px; color:#64748b;">{fmt_lakh(result['max_pe_oi'])} contracts</div>
        </div>
        """, unsafe_allow_html=True)

        # OI Buildup Bar
        total_bu = result["ce_buildup"] + result["pe_buildup"]
        ce_pct = (result["ce_buildup"] / total_bu * 100) if total_bu > 0 else 50
        pe_pct = 100 - ce_pct

        st.markdown(f"""
        <div class="info-box">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <div>
                    <span style="font-size:11px; color:#f43f5e;">CE OI Added: </span>
                    <span style="font-size:14px; font-weight:700; font-family:'JetBrains Mono'; color:#e2e8f0;">{fmt_lakh(result['ce_buildup'])}</span>
                    <span style="font-size:11px; color:#64748b;"> ({ce_pct:.0f}%)</span>
                </div>
                <div>
                    <span style="font-size:11px; color:#22c55e;">PE OI Added: </span>
                    <span style="font-size:14px; font-weight:700; font-family:'JetBrains Mono'; color:#e2e8f0;">{fmt_lakh(result['pe_buildup'])}</span>
                    <span style="font-size:11px; color:#64748b;"> ({pe_pct:.0f}%)</span>
                </div>
            </div>
            <div class="buildup-bar">
                <div class="buildup-ce" style="width:{ce_pct}%;"></div>
                <div class="buildup-pe" style="width:{pe_pct}%;"></div>
            </div>
            <div style="font-size:10px; color:#64748b; margin-top:4px;">
                {'⬆ More PE buildup → Support forming → Bullish' if pe_pct > 55 else '⬇ More CE buildup → Resistance forming → Bearish' if ce_pct > 55 else '⚖ Balanced buildup → Range-bound'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Option Chain Table ──
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <span style="font-size:16px; font-weight:600; color:#e2e8f0;">Option Chain — {selected_index}</span>
        <span style="font-size:12px; color:#64748b;">{len(display_df)} strikes around ATM</span>
    </div>
    """, unsafe_allow_html=True)

    # Build styled HTML table with ATM row highlight
    max_oi = max(display_df["ce_oi"].max(), display_df["pe_oi"].max(), 1)

    headers = ["CE OI", "CE Chg", "CE Vol", "CE LTP", "CE IV%", "★ STRIKE", "PE IV%", "PE LTP", "PE Vol", "PE Chg", "PE OI"]
    html = """
    <div style="overflow-x:auto; border:1px solid #1e293b; border-radius:12px;">
    <table style="width:100%; border-collapse:collapse; font-family:'JetBrains Mono',monospace; font-size:12px;">
    <thead><tr>"""
    for h in headers:
        strike_style = "background:#38bdf811; color:#38bdf8; font-size:13px;" if "STRIKE" in h else ""
        html += f'<th style="padding:10px 8px; text-align:right; color:#64748b; font-size:10px; letter-spacing:1px; font-weight:600; border-bottom:1px solid #1e293b; background:#111827; {strike_style}">{h}</th>'
    html += "</tr></thead><tbody>"

    for _, r in display_df.iterrows():
        is_atm = r["strike"] == atm
        ce_chg_sign = "+" if r["ce_oi_chg"] > 0 else ""
        pe_chg_sign = "+" if r["pe_oi_chg"] > 0 else ""
        ce_chg_color = "#f43f5e" if r["ce_oi_chg"] > 0 else "#22c55e" if r["ce_oi_chg"] < 0 else "#64748b"
        pe_chg_color = "#22c55e" if r["pe_oi_chg"] > 0 else "#f43f5e" if r["pe_oi_chg"] < 0 else "#64748b"

        # ATM row styling
        if is_atm:
            row_bg = "background: linear-gradient(90deg, #f43f5e11 0%, #38bdf822 50%, #22c55e11 100%);"
            row_border = "border-top:2px solid #38bdf8; border-bottom:2px solid #38bdf8;"
        else:
            row_bg = "background:transparent;"
            row_border = "border-bottom:1px solid #1e293b33;"

        html += f'<tr style="{row_bg} {row_border}">'

        # CE OI with bar
        ce_pct = min((r["ce_oi"] / max_oi) * 100, 100) if max_oi > 0 else 0
        html += f'''<td style="padding:8px; text-align:right; color:#e2e8f0;">
            {fmt_lakh(r["ce_oi"])}
            <div style="width:100%;height:4px;background:#1e293b;border-radius:2px;margin-top:3px;">
                <div style="width:{ce_pct}%;height:100%;background:#f43f5e;border-radius:2px;"></div>
            </div>
        </td>'''

        # CE Chg
        html += f'<td style="padding:8px; text-align:right; color:{ce_chg_color};">{ce_chg_sign}{fmt_lakh(r["ce_oi_chg"])}</td>'

        # CE Vol
        html += f'<td style="padding:8px; text-align:right; color:#94a3b8;">{fmt_lakh(r["ce_volume"])}</td>'

        # CE LTP
        html += f'<td style="padding:8px; text-align:right; color:#e2e8f0;">{r["ce_ltp"]:.2f}</td>'

        # CE IV
        html += f'<td style="padding:8px; text-align:right; color:#94a3b8;">{r["ce_iv"]:.1f}%</td>'

        # STRIKE (center column)
        if is_atm:
            strike_cell_style = "padding:8px 12px; text-align:center; font-weight:700; font-size:14px; color:#38bdf8; background:#38bdf822; text-shadow:0 0 8px #38bdf844;"
            atm_badge = '<div style="font-size:8px; letter-spacing:3px; color:#38bdf8; margin-top:2px;">◆ ATM ◆</div>'
            spot_line = f'<div style="font-size:9px; color:#ffd740; margin-top:1px;">SPOT {spot:,.2f}</div>'
        else:
            strike_cell_style = "padding:8px 12px; text-align:center; font-weight:600; font-size:13px; color:#e2e8f0;"
            atm_badge = ""
            spot_line = ""
        html += f'<td style="{strike_cell_style}">{int(r["strike"]):,}{atm_badge}{spot_line}</td>'

        # PE IV
        html += f'<td style="padding:8px; text-align:right; color:#94a3b8;">{r["pe_iv"]:.1f}%</td>'

        # PE LTP
        html += f'<td style="padding:8px; text-align:right; color:#e2e8f0;">{r["pe_ltp"]:.2f}</td>'

        # PE Vol
        html += f'<td style="padding:8px; text-align:right; color:#94a3b8;">{fmt_lakh(r["pe_volume"])}</td>'

        # PE Chg
        html += f'<td style="padding:8px; text-align:right; color:{pe_chg_color};">{pe_chg_sign}{fmt_lakh(r["pe_oi_chg"])}</td>'

        # PE OI with bar
        pe_pct = min((r["pe_oi"] / max_oi) * 100, 100) if max_oi > 0 else 0
        html += f'''<td style="padding:8px; text-align:right; color:#e2e8f0;">
            {fmt_lakh(r["pe_oi"])}
            <div style="width:100%;height:4px;background:#1e293b;border-radius:2px;margin-top:3px;">
                <div style="width:{pe_pct}%;height:100%;background:#22c55e;border-radius:2px;"></div>
            </div>
        </td>'''

        html += "</tr>"

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

    # ── Visual OI Comparison Chart ──
    st.markdown("---")
    st.markdown('<div style="font-size:16px; font-weight:600; color:#e2e8f0; margin-bottom:12px;">CE vs PE — Open Interest Distribution</div>', unsafe_allow_html=True)

    chart_data = display_df[["strike", "ce_oi", "pe_oi"]].copy()
    chart_data.columns = ["Strike", "Call OI", "Put OI"]
    chart_data["Strike"] = chart_data["Strike"].astype(int).astype(str)
    chart_data = chart_data.set_index("Strike")

    st.bar_chart(chart_data, color=["#f43f5e", "#22c55e"], height=350)

    # ── Methodology ──
    with st.expander("📘 Scoring Methodology & Logic"):
        st.markdown("""
        ### 4-Factor Composite Directional Model

        | Factor | Weight | Bullish Signal | Bearish Signal |
        |--------|--------|---------------|----------------|
        | **PCR (OI)** | 30% | > 1.3 (heavy put writing = support) | < 0.7 (heavy call writing = resistance) |
        | **Max OI Walls** | 25% | Resistance far from ATM | Support far from ATM |
        | **OI Buildup** | 25% | PE additions > 65% | CE additions > 65% |
        | **Volume PCR** | 20% | > 1.2 (put volume dominance) | < 0.8 (call volume dominance) |

        **Score Thresholds:**
        - **±40+**: Strong signal (high conviction)
        - **±20–39**: Moderate signal (watch for confirmation)
        - **±5–19**: Mild signal (low conviction)
        - **-4 to +4**: Neutral / range-bound

        **Key Concepts:**
        - **Put writing at a strike** = seller believes price will STAY ABOVE that strike → acts as SUPPORT
        - **Call writing at a strike** = seller believes price will STAY BELOW that strike → acts as RESISTANCE
        - **OI buildup momentum** shows where fresh money is flowing — this is the "smart money" factor
        - **Volume PCR** confirms whether OI positions are being actively reinforced
        """)

    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

except requests.exceptions.HTTPError as e:
    if "401" in str(e) or "403" in str(e):
        st.error("🔒 **Authentication Failed** — Your token may have expired. Upstox tokens expire daily. Please generate a new one.")
    else:
        st.error(f"⚠️ **API Error:** {e}")
except requests.exceptions.ConnectionError:
    st.error("🌐 **Connection Error** — Could not reach Upstox API. Check your internet connection.")
except Exception as e:
    st.error(f"⚠️ **Error:** {str(e)}")
    st.exception(e)
