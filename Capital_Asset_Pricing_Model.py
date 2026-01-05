import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Risk-free rate Romania (~7% randamentul titlurilor de stat 10Y)
RISK_FREE_RATE = 0.07
MONTHS_IN_YEAR = 12


class CAPM:
    def __init__(self, stock_ticker, bet_csv):
        self.stock_ticker = stock_ticker
        self.bet_csv = bet_csv
        self.data = None

    # 1. Download TRP data from yfinance
    def load_stock(self):
        print(f"Downloading {self.stock_ticker} from yfinance...")

        try:
            stock = yf.Ticker(self.stock_ticker)
            df = stock.history(period="max")
        except Exception as e:
            print("Error downloading TRP:", e)
            return None

        if df.empty:
            raise ValueError("No data returned for TRP.RO")

        # Use Close or Adj Close
        if "Adj Close" in df.columns:
            df = df[["Adj Close"]].rename(columns={"Adj Close": "stock"})
        else:
            df = df[["Close"]].rename(columns={"Close": "stock"})

        return df

    # 2. Load BET from BVB CSV
    def load_bet_csv(self):
        print(f"Loading BET from: {self.bet_csv}")

        # BET are separatorul ";"
        df = pd.read_csv(self.bet_csv, sep=";")

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Detect date column
        date_col = [c for c in df.columns if "data" in c or "date" in c][0]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df = df.set_index(date_col)

        # Detect price ("pret")
        price_col = [c for c in df.columns if "pret" in c][0]
        df = df[[price_col]].rename(columns={price_col: "bet"})

        #formatul romanesc excel "22860,42" → 22860.42
        df["bet"] = (
            df["bet"]
            .astype(str)
            .str.replace(".", "", regex=False)  # remove thousands separator
            .str.replace(",", ".", regex=False)  # decimal comma → dot
            .str.strip()
        )
        df["bet"] = pd.to_numeric(df["bet"], errors="coerce")

        # Remove NaN
        df = df.dropna()

        # Remove duplicates & sort
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        return df

    # 3. Merge TRP + BET and create monthly returns
    def initialize(self):
        trp = self.load_stock()
        bet = self.load_bet_csv()

        #remove timezone from TRP (tz-aware → tz-naive)
        if hasattr(trp.index, 'tz'):
            trp.index = trp.index.tz_localize(None)

        bet.index = pd.to_datetime(bet.index)
        #merge
        df = pd.concat([trp, bet], axis=1)
        df = df.resample("M").last()

        df["trp_ret"] = np.log(df["stock"] / df["stock"].shift(1))
        df["bet_ret"] = np.log(df["bet"] / df["bet"].shift(1))

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        if len(df) < 100:
            raise ValueError("Not enough data for CAPM.")

        self.data = df

        print("\nFinal dataset (TRP + BET monthly returns):")
        print(df.head())
        print(df.tail())

    # 4. Beta (covariance)
    def beta_covariance(self):
        cov = np.cov(self.data["trp_ret"], self.data["bet_ret"])
        beta = cov[0, 1] / cov[1, 1]
        print(f"\nBeta (covariance): {beta:.4f}")
        return beta

    # 5. Regression (beta + alpha + expected return CAPM)
    def regression(self):
        x = self.data["bet_ret"]
        y = self.data["trp_ret"]

        beta, alpha = np.polyfit(x, y, 1)

        print(f"Beta (regression): {beta:.4f}")
        print(f"Alpha: {alpha:.4f}")

        annual_bet_return = self.data["bet_ret"].mean() * MONTHS_IN_YEAR
        expected_return = RISK_FREE_RATE + beta * (annual_bet_return - RISK_FREE_RATE)

        print(f"Expected return (CAPM): {expected_return:.4f}")

        return beta, alpha, expected_return

    # 6. Plot CAPM regression
    def plot(self, beta, alpha):
        plt.figure(figsize=(12, 8))
        plt.scatter(self.data["bet_ret"], self.data["trp_ret"], alpha=0.5, label="Data Points")
        plt.plot(self.data["bet_ret"], beta * self.data["bet_ret"] + alpha,
                 color="red", label="CAPM Regression Line")

        plt.title("CAPM Regression: TRP.RO vs BET Index")
        plt.xlabel("BET Returns (Rm)")
        plt.ylabel("TRP Returns (Ra)")
        plt.grid(True)
        plt.legend()
        plt.show()


# Run
if __name__ == "__main__":
    capm = CAPM("TRP.RO", "Bet.csv")
    capm.initialize()
    beta_cov = capm.beta_covariance()
    beta, alpha, exp_ret = capm.regression()
    capm.plot(beta, alpha)
