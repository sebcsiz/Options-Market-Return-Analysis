import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(67)

BLUE   = "#2E5090"
ORANGE = "#E87722"
GREEN  = "#2E8B57"
RED    = "#C0392B"
GRAY   = "#888888"
PURPLE = "#7B2D8B"

plt.rcParams.update({
      "font.family"      : "serif",
      "axes.spines.top"  : False,
      "axes.spines.right": False,
      "axes.titlesize"   : 13,
      "axes.labelsize"   : 11,
      "figure.dpi"       : 150,
})

PRIOR_MU   = 0.0
PRIOR_TAU2 = 100.0


def load_data(filepath="data/bendgame/options.csv"):
      df = pd.read_csv(filepath)
      df['Date']           = pd.to_datetime(df['Date'])
      df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'])

      df['SignedMoneyness'] = np.where(
            df['OptionType'] == 'call',
            (df['StockPrice'] - df['Strike']) / df['Strike'] * 100,
            (df['Strike']     - df['StockPrice']) / df['Strike'] * 100,
      )
      
      p1, p99  = df['SignedMoneyness'].quantile([0.01, 0.99])
      df_clean = df[(df['SignedMoneyness'] >= p1) &
                        (df['SignedMoneyness'] <= p99)].copy()

      df_clean['DTE']      = (df_clean['ExpirationDate'] - df_clean['Date']).dt.days
      df_clean['LogSpent'] = np.log(df_clean['Spent'])

      returns = df_clean['SignedMoneyness'].values
      return df, df_clean, returns


def print_summary(df_raw, df_clean, returns):
      n = len(returns)
      print("=" * 65)
      print("STAT 401 — Options Market Trades")
      print("=" * 65)
      print(f"  Raw trades                    : {len(df_raw):,}")
      print(f"  After 1st/99th pct trim       : {n:,}  "
            f"(dropped {len(df_raw)-n:,})")
      print(f"  Unique trading days           : {df_clean['Date'].nunique()}")
      print(f"  Unique symbols                : {df_clean['Sym'].nunique()}")
      print(f"  Date range                    : "
            f"{df_clean['Date'].min().date()} to "
            f"{df_clean['Date'].max().date()}")
      print(f"  Calls / Puts                  : "
            f"{(df_clean['OptionType']=='call').sum():,} / "
            f"{(df_clean['OptionType']=='put').sum():,}")
      print(f"  ITM / ATM / OTM               : "
            f"{(df_clean['ChainLocation']=='ITM').sum():,} / "
            f"{(df_clean['ChainLocation']=='ATM').sum():,} / "
            f"{(df_clean['ChainLocation']=='OTM').sum():,}")
      print(f"  SignedMoneyness mean          : {returns.mean():.4f}%")
      print(f"  SignedMoneyness std           : {returns.std(ddof=1):.4f}%")
      print(f"  Skewness                      : {stats.skew(returns):.3f}")
      print(f"  Excess kurtosis               : {stats.kurtosis(returns):.3f}")