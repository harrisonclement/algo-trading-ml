"""
ProTrader AI — Streamlit dashboard.

RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run ProTrader7.py

MODEL:
    This app loads a trained model from the local ./model directory (an MLflow
    model artifact bundled in the repo). There are NO credentials here and no
    network call at runtime — the model is served straight from disk.

    To (re)create ./model, run export_model.py once. See that file for details.
"""

import streamlit as st
from pathlib import Path
import mlflow.pyfunc
import yfinance as yf
import pandas as pd
import numpy as np

# ======================================================================
# 1. STREAMLIT CONFIGURATION
# ======================================================================
st.set_page_config(page_title="ProTrader AI", layout="wide")

# ======================================================================
# 2. MODEL CONFIG  (no credentials — model is loaded from disk)
# ======================================================================
# Path is resolved relative to this file, so it works no matter what the
# current working directory is (local run, Streamlit Cloud, container, etc.).
MODEL_PATH = Path(__file__).parent / "model"

# Cosmetic label only — shown in the sidebar. Not used to fetch anything.
MODEL_ID = "6596bfdef96840f5bc13b511f7cfe114"

# ======================================================================
# 3. HELPER FUNCTIONS
# ======================================================================
@st.cache_resource
def load_predictor(model_path: Path):
    """Loads the bundled MLflow model from the local filesystem."""
    try:
        if not model_path.exists():
            st.error(
                f"❌ Model not found at '{model_path}'. "
                "Run `python export_model.py` to bundle the model into the repo."
            )
            return None
        return mlflow.pyfunc.load_model(str(model_path))
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def get_market_data(ticker):
    """
    Fetches data and generates the EXACT schema required by the model.
    Also returns company metadata (Name, Sector).
    """
    try:
        # 1. Get Metadata (Name, Sector)
        t = yf.Ticker(ticker)
        try:
            info = t.info
            meta = {
                'name': info.get('shortName', ticker),
                'sector': info.get('sector', 'Unknown Sector'),
                'price': info.get('currentPrice', None)
            }
        except Exception:
            meta = {'name': ticker, 'sector': '-', 'price': None}

        # 2. Fetch History (6 months)
        df = yf.download(ticker, period="6mo", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return None, None, None

        # FLATTEN MULTI-INDEX COLUMNS
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- FEATURE ENGINEERING ---
        df['Return_Lag1'] = df['Close'].pct_change(1)
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df = df.dropna()
        if df.empty:
            return None, None, None

        # --- PREPARE INPUTS ---
        required_features = ['Return_Lag1', 'MA_10', 'MA_50', 'Volatility_10', 'Momentum']
        latest_features = df.iloc[[-1]][required_features]
        return latest_features, df, meta

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

# ======================================================================
# 4. MAIN APP UI
# ======================================================================
st.markdown("""
<style>
.metric-card {
    background-color: #0E1117;
    border: 1px solid #303030;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.signal-box {
    padding: 40px;
    border-radius: 8px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="NFLX").upper()
st.sidebar.divider()
st.sidebar.subheader("🎚️ Risk Management")
threshold_bps = st.sidebar.slider(
    "Minimum Conviction (Basis Points)",
    min_value=0,
    max_value=50,
    value=10,
    step=5,
    help="Filter out weak signals. 10 bps = 0.10% predicted move."
)
threshold_decimal = threshold_bps / 10000
st.sidebar.caption(f"Model ID: {MODEL_ID}")

# --- USER INSTRUCTIONS (FRONTEND) ---
with st.expander("📘 User Guide: How to use ProTrader", expanded=True):
    st.markdown("""
    ### Welcome to ProTrader AI
    This dashboard uses a Random Forest regression model to forecast the
    next day's price return, then converts that forecast into a signal.

    **Quick Start:**
    1. **Select Asset:** Enter a ticker symbol in the sidebar (e.g., `NVDA`, `SPY`, `TSLA`).
    2. **Set Conviction:** Use the **Risk Management** slider in the sidebar.
        * *Higher values (e.g., 20)* = Safer. The model only signals if it predicts a large move.
        * *Lower values (e.g., 5)* = More Aggressive.
    3. **Read Signals:**
        * <span style='color:#1B5E20; font-weight:bold'>BUY</span>: Forecast return above your threshold.
        * <span style='color:#B71C1C; font-weight:bold'>SELL</span>: Forecast return below the negative threshold.
        * <span style='color:#808080; font-weight:bold'>HOLD</span>: Forecast too weak (inside your conviction band).
    """, unsafe_allow_html=True)

# --- MAIN PAGE LOGIC ---
predictor = load_predictor(MODEL_PATH)

if predictor:
    X_live, full_df, meta = get_market_data(ticker)

    if X_live is not None:
        # 1. HEADER SECTION
        st.divider()
        col_head1, col_head2 = st.columns([3, 1])
        with col_head1:
            st.markdown(f"# {ticker} | {meta['name']}")
            st.markdown(f"**Sector:** {meta['sector']}")
        with col_head2:
            current_price = full_df['Close'].iloc[-1]
            st.metric("Latest Close", f"${current_price:.2f}")

        st.divider()

        # 2. SIGNAL GENERATION
        pred_return = predictor.predict(X_live)[0]

        if pred_return > threshold_decimal:
            signal = "BUY"
            bg_color = "#1B5E20"
            sub_text = "Model forecasts upward continuation > threshold"
        elif pred_return < -threshold_decimal:
            signal = "SELL"
            bg_color = "#B71C1C"
            sub_text = "Model forecasts downward break > threshold"
        else:
            signal = "HOLD"
            bg_color = "#424242"
            sub_text = f"Forecast ({pred_return*10000:.1f} bps) below conviction threshold"

        # 3. HERO SIGNAL CARD
        st.markdown(f"""
        <div class="signal-box" style="background-color: {bg_color};">
            <h1 style="font-size: 60px; margin:0; font-weight: 700; letter-spacing: 4px;">{signal}</h1>
            <p style="font-size: 18px; margin-top: 10px; opacity: 0.9;">{sub_text}</p>
        </div>
        """, unsafe_allow_html=True)

        # 4. ANALYTICS ROW
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("##### Predicted Return (T+1)")
            st.markdown(f"## {pred_return:.4%}")
            st.caption("Model's raw output")
        with m2:
            st.markdown("##### Risk Threshold")
            st.markdown(f"## +/- {threshold_decimal:.4%}")
            st.caption("Your setting from sidebar")
        with m3:
            st.markdown("##### Trend Context")
            ma_50 = full_df['MA_50'].iloc[-1]
            trend = "Uptrend" if current_price > ma_50 else "Downtrend"
            trend_color = "green" if trend == "Uptrend" else "red"
            st.markdown(f"## :{trend_color}[{trend}]")
            st.caption("Vs. 50-Day Moving Average")

        # 5. CHART SECTION
        st.write("")
        st.write("")
        st.subheader("Technical Chart")
        chart_data = full_df[['Close', 'MA_10', 'MA_50']].tail(120)
        st.line_chart(chart_data, color=["#E0E0E0", "#00C805", "#FF5000"], height=400)

        # 6. EXPANDABLE DATA
        with st.expander("🔍 View Raw Model Inputs"):
            st.dataframe(X_live)
    else:
        st.warning(f"Could not load data for {ticker}. Please check the symbol.")
else:
    st.info("Model not loaded. See the error above.")
