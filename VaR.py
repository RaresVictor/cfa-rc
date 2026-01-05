import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t
#Descarca preturile pentru TRP de pe Yahoo Finance si calculeaza randamentele simple zilnice, impreuna cu ultimul pret disponibil.
#Calculeaza VaR prin metoda Historical-Bootstrap, simuland evolutia pretului prin resampling cu inlocuire din randamentele istorice.
#Calculeaza VaR folosind o simulare Monte Carlo bazata pe distributia Student-t, ajustand parametrii distributiei la randamentele istorice.
#Genereaza un numar mare de scenarii Monte Carlo, simuleaza evolutia pretului pe orizontul dorit si estimeaza pierderea maxima la nivelul de incredere specificat.
#Afiseaza VaR-ul istoric, VaR-ul Student-t si parametrii distributiei ajustate.

def fetch_returns_and_price(ticker, start=None, end=None):
    data = yf.download(ticker, start=start, end=end,
                       progress=False, auto_adjust=False)

    prices = data["Close"].dropna()

    returns = prices.pct_change().dropna()

    #current_price is a scalar float
    current_price = prices.iloc[-1].item()

    return returns, current_price



def bootstrap_var(returns,
                  current_price=None,
                  num_simulations=10_000,
                  horizon_days=1,
                  confidence=0.95):
    """
    Historical-bootstrap Monte Carlo VaR:
    - Resamples historical returns with replacement
    - Simulates horizon_days ahead
    """
    if current_price is None:
        current_price = 1.0

    # Ensure 1-D numpy array
    r = np.asarray(returns).flatten()

    sampled = np.random.choice(r,
                               size=(num_simulations, horizon_days),
                               replace=True)

    cumulative_return = np.prod(1 + sampled, axis=1) - 1
    simulated_prices = current_price * (1 + cumulative_return)

    pnl = simulated_prices - current_price
    var_value = -np.percentile(pnl, (1 - confidence) * 100)
    return var_value


def student_t_var(returns,
                  current_price=None,
                  num_simulations=10_000,
                  horizon_days=1,
                  confidence=0.95):
    """
    Student-t Monte Carlo VaR:
    - Fit Student-t distribution to returns
    - Simulate horizon_days returns from fitted distribution
    """
    if current_price is None:
        current_price = 1.0

    # Ensure 1-D numpy array
    r = np.asarray(returns).flatten()

    # Fit Student-t distribution (df, mean/loc, scale)
    df, loc, scale = t.fit(r)

    simulated = t.rvs(df=df, loc=loc, scale=scale,
                      size=(num_simulations, horizon_days))

    cumulative_return = np.prod(1 + simulated, axis=1) - 1
    simulated_prices = current_price * (1 + cumulative_return)

    pnl = simulated_prices - current_price
    var_value = -np.percentile(pnl, (1 - confidence) * 100)
    return var_value, df, loc, scale


if __name__ == "__main__":
    np.random.seed(42)

    ticker = "TRP.RO"

    # Fetch data and compute returns + last price
    returns, last_price = fetch_returns_and_price(
        ticker,
        start="2020-01-01",
        end=None
    )

    # Historical-bootstrap VaR
    hb_var = bootstrap_var(
        returns,
        current_price=last_price,
        num_simulations=50_000,
        horizon_days=1,
        confidence=0.95
    )

    # Student-t VaR
    t_var, df, loc, scale = student_t_var(
        returns,
        current_price=last_price,
        num_simulations=50_000,
        horizon_days=1,
        confidence=0.95
    )

    print(f"Ticker: {ticker}")
    print(f"Last price: {last_price:.2f}")
    print(f"Historical-Bootstrap Monte Carlo VaR (95%): {hb_var:.4f} RON per Share")
    print(f"Student-t Monte Carlo VaR (95%):        {t_var:.4f} RON per Share")
    print(f"Fitted Student-t params: df={df:.2f}, loc={loc:.6f}, scale={scale:.6f}")
