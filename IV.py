import streamlit as st
import yfinance as yf
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import date

# -------------------------
# Black-Scholes & IV function
# -------------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)

    if option_type == "call":
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def implied_volatility(option_price, S, K, T, r, option_type="call"):
    try:
        return brentq(
            lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - option_price,
            1e-6, 5
        )
    except:
        return np.nan

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Implied Volatility Viewer", layout="wide")
st.title("📈 Apple Options Implied Volatility Viewer")

ticker = "AAPL"
stock = yf.Ticker(ticker)
S = stock.history(period="1d")["Close"].iloc[-1]
st.write(f"**Current {ticker} price:** ${S:.2f}")

expiries = stock.options
expiry = st.selectbox("Choose expiry date:", expiries)

option_type = st.radio("Choose option type:", ["call", "put"])

opt_chain = stock.option_chain(expiry)
options = opt_chain.calls if option_type == "call" else opt_chain.puts

# Time to expiry
T = (date.fromisoformat(expiry) - date.today()).days / 365
r = 0.045  # Example: 10y yield

options["mid"] = (options["bid"] + options["ask"]) / 2
options["implied_vol"] = options.apply(
    lambda row: implied_volatility(row["mid"], S, row["strike"], T, r, option_type),
    axis=1
)

st.dataframe(options[["strike", "mid", "implied_vol"]])