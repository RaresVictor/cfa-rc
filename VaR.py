import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t

np.random.seed(11)
#parametrii
ticker = "TRP.RO"
start_date = "2010-01-01"
confidence = 0.95
num_simulations = 50_000
horizon_days = 1

try:
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")
except Exception as e:
    print("Error downloading", e)
    df = None

if df is None or df.empty:
    raise ValueError(f"No data returned for {ticker}")

if "Adj Close" in df.columns:
    df = df[["Adj Close"]].rename(columns={"Adj Close": "stock"})
else:
    df = df[["Close"]].rename(columns={"Close": "stock"})

# Use adjusted prices for VaR
prices = df["stock"].dropna()
returns = prices.pct_change().dropna()

current_price = float(prices.iloc[-1])
r = returns.values

sampled = np.random.choice(r, size=(num_simulations, horizon_days), replace=True)

cumulative_return = np.prod(1 + sampled, axis=1) - 1
simulated_prices = current_price * (1 + cumulative_return)
pnl = simulated_prices - current_price

hb_var = -np.percentile(pnl, (1 - confidence) * 100)

# Student-t Monte Carlo VaR

df, loc, scale = t.fit(r)

simulated_t = t.rvs(df=df,loc=0,scale=scale,size=(num_simulations, horizon_days))

cumulative_return_t = np.prod(1 + simulated_t, axis=1) - 1
simulated_prices_t = current_price * (1 + cumulative_return_t)
pnl_t = simulated_prices_t - current_price

t_var = -np.percentile(pnl_t, (1 - confidence) * 100)

# Output

print(f"Last price: {current_price:.4f}")
print(f"Historical Bootstrap VaR (95%): {hb_var:.4f} RON")
print(f"Student-t Monte Carlo VaR (95%): {t_var:.4f} RON")
#print(f"Student-t parameters: df={df:.2f}, loc={loc:.6f}, scale={scale:.6f}")
