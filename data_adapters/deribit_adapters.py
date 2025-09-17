import pandas as pd
import numpy as np

def read_parquet(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pandas()

def clean_deribit(df: pd.DataFrame, reference_date="2025-01-30", r=0.0, q=0.0) -> pd.DataFrame:
    """
    - Splits instrument_name into asset/expiry/strike/type
    - Builds τ (in years, 365*24*3600 continuous trading)
    - Uses Deribit mid_price which is *already quoted in units of the underlying*.
      We convert to normalized price c = C/F. Since mid_price = C/S, then:
          c = (C/F) = (C/S) * (S/F) = mid_price * exp(-(r - q) * τ).
    """
    df = df.copy()
    df[['asset', 'expiry', 'strike', 'option_type']] = df['instrument_name'].str.split('-', expand=True)
    df['expiry'] = pd.to_datetime(df['expiry'])
    ref = pd.to_datetime(reference_date)
    df['tau'] = (df['expiry'] - ref).dt.days / 365.25

    # Only calls with positive USD volume (to bias toward liquid quotes)
    df = df[(df['option_type'] == 'C') & (df['stats_volume_usd'] > 0)]

    # numeric strike, forward, log‑moneyness
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    # mid_price already in units of underlying (C/S)
    df['mid_price'] = (df['best_bid_price'] + df['best_ask_price']) / 2.0

    # forward and normalized call price c = C/F
    df['F'] = df['underlying_price'] * np.exp((r - q) * df['tau'])
    df['m'] = np.log(df['strike'] / df['F'])
    df['c_norm'] = df['mid_price'] * np.exp(-(r - q) * df['tau'])  # (C/S)*S/F = C/F

    # Timestamps to pandas datetime (Deribit is ms since epoch typically)
    if np.issubdtype(df['timestamp'].dtype, np.number):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop nonsense
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['tau', 'm', 'c_norm', 'underlying_price'])
    return df
