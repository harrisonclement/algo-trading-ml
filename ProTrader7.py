"""
⬇️⬇️⬇️ INSTRUCTIONS TO RUN (FOR DEVELOPER) ⬇️⬇️⬇️

1.  Open your terminal.
2.  Install libraries: pip install streamlit mlflow yfinance pandas numpy databricks-sdk
3.  Run the app: streamlit run app.py

⬆️⬆️⬆️ END DEV INSTRUCTIONS ⬆️⬆️⬆️
"""

import streamlit as st
import os
import mlflow.pyfunc
import yfinance as yf
import pandas as pd
import numpy as np

# ======================================================================
# 1. STREAMLIT CONFIGURATION
# ======================================================================
st.set_page_config(page_title="ProTrader AI", layout="wide")

# ======================================================================
# 2. CREDENTIALS & MODEL CONFIG
# ======================================================================

# 🚨 RUN ID (This ID requires Pro Features: MAs, Volatility, Momentum)
BEST_RUN_ID = "6596bfdef96840f5bc13b511f7cfe114"

# 🔑 DATABRICKS CREDENTIALS
os.environ["DATABRICKS_HOST"] = "https://dbc-b5e97478-b957.cloud.databricks.com/"
os.environ["DATABRICKS_TOKEN"] = "dapi2806faac61e06eef37ff4a4f4a0d8f3b"

# ======================================================================
# 3. HELPER FUNCTIONS
# ======================================================================

@st.cache_resource
def load_predictor(run_id):
    """Loads the model from Databricks MLflow."""
    try:
        model_uri = f"runs:/{run_id}/model"
        mlflow.set_tracking_uri("databricks")
        return mlflow.pyfunc.load_model(model_uri)
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
        except:
            meta = {'name': ticker, 'sector': '-', 'price': None}

        # 2. Fetch History (6 months)
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
        
        if df.empty:
            return None, None, None
            
        # 🚨 FIX: FLATTEN MULTI-INDEX COLUMNS
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

# --- CSS STYLING FOR PROFESSIONAL LOOK ---
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

st.sidebar.caption(f"Model ID: {BEST_RUN_ID}")

# --- 🆕 USER INSTRUCTIONS (FRONTEND) ---
with st.expander("📘 User Guide: How to use ProTrader", expanded=True):
    st.markdown("""
    ### Welcome to ProTrader AI
    This professional dashboard uses a Random Forest model to forecast short-term price direction.
    
    **Quick Start:**
    1.  **Select Asset:** Enter a ticker symbol in the sidebar (e.g., `NVDA`, `SPY`, `TSLA`).
    2.  **Set Conviction:** Use the **Risk Management** slider in the sidebar.
        * *Higher values (e.g., 20)* = Safer. The model only signals if it predicts a large move.
        * *Lower values (e.g., 5)* = More Aggressive.
    3.  **Read Signals:**
        * <span style='color:#1B5E20; font-weight:bold'>BUY</span>: Model predicts strong upward momentum.
        * <span style='color:#B71C1C; font-weight:bold'>SELL</span>: Model predicts strong downward momentum.
        * <span style='color:#808080; font-weight:bold'>HOLD</span>: Signal is too weak (below your conviction threshold).
    """, unsafe_allow_html=True)

# --- MAIN PAGE LOGIC ---
predictor = load_predictor(BEST_RUN_ID)

if predictor:
    # Fetch Data
    X_live, full_df, meta = get_market_data(ticker)
    
    if X_live is not None:
        
        # ---------------------------------------------------------
        # 1. HEADER SECTION (Ticker | Name | Sector)
        # ---------------------------------------------------------
        st.divider() # Separator after instructions
        
        col_head1, col_head2 = st.columns([3, 1])
        with col_head1:
            st.markdown(f"# {ticker} | {meta['name']}")
            st.markdown(f"**Sector:** {meta['sector']}")
        with col_head2:
            current_price = full_df['Close'].iloc[-1]
            st.metric("Latest Close", f"${current_price:.2f}")

        st.divider()

        # ---------------------------------------------------------
        # 2. SIGNAL GENERATION
        # ---------------------------------------------------------
        pred_return = predictor.predict(X_live)[0]
        
        # Logic
        if pred_return > threshold_decimal:
            signal = "BUY"
            bg_color = "#1B5E20"  # Professional Dark Green
            sub_text = "Model forecasts upward continuation > threshold"
        elif pred_return < -threshold_decimal:
            signal = "SELL"
            bg_color = "#B71C1C"  # Professional Dark Red
            sub_text = "Model forecasts downward break > threshold"
        else:
            signal = "HOLD"
            bg_color = "#424242"  # Professional Dark Grey
            sub_text = f"Forecast ({pred_return*10000:.1f} bps) below conviction threshold"

        # ---------------------------------------------------------
        # 3. HERO SIGNAL CARD
        # ---------------------------------------------------------
        st.markdown(f"""
        <div class="signal-box" style="background-color: {bg_color};">
            <h1 style="font-size: 60px; margin:0; font-weight: 700; letter-spacing: 4px;">{signal}</h1>
            <p style="font-size: 18px; margin-top: 10px; opacity: 0.9;">{sub_text}</p>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 4. ANALYTICS ROW
        # ---------------------------------------------------------
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
            # Simple logic: Is price above 50-day MA?
            ma_50 = full_df['MA_50'].iloc[-1]
            trend = "Uptrend" if current_price > ma_50 else "Downtrend"
            trend_color = "green" if trend == "Uptrend" else "red"
            st.markdown(f"## :{trend_color}[{trend}]")
            st.caption("Vs. 50-Day Moving Average")

        # ---------------------------------------------------------
        # 5. CHART SECTION
        # ---------------------------------------------------------
        st.write("")
        st.write("")
        st.subheader("Technical Chart")
        
        # Clean chart with Price, MA10, MA50
        chart_data = full_df[['Close', 'MA_10', 'MA_50']].tail(120)
        st.line_chart(chart_data, color=["#E0E0E0", "#00C805", "#FF5000"], height=400)
        
        # ---------------------------------------------------------
        # 6. EXPANDABLE DATA
        # ---------------------------------------------------------
        with st.expander("🔍 View Raw Model Inputs"):
            st.dataframe(X_live)

    else:
        st.warning(f"Could not load data for {ticker}. Please check the symbol.")

else:
    st.info("Connecting to MLflow Model Registry...")
