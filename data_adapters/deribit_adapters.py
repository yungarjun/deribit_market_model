import pandas as pd
import numpy as np
import json
from pandas import json_normalize
from config import MicrostructureCfg
from typing import Optional
from utils.black_scholes import implied_vol_from_c_norm, bs_vega_norm

def read_parquet(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pandas()

def clean_deribit(df: pd.DataFrame, reference_date, r=0.0, q=0.0,
                  micro: Optional[MicrostructureCfg] = None ) -> pd.DataFrame:
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

    # Apply Microstructure filters if requested
    if micro is not None:
        df = apply_microstructure_filters(df, micro)

    return df



def apply_microstructure_filters(df: pd.DataFrame, micro: MicrostructureCfg) -> pd.DataFrame:
    """
    A1 microstructure filters, applied before any binning/K-means.
    """
    dbg = getattr(micro, 'debug', False)
    def log(step, d):
        if dbg:
            print(f"[micro] {step:<22} -> rows={len(d)}, times={d['timestamp'].nunique() if 'timestamp' in d else 'na'}, taus={d['tau'].nunique()}")

    df = df.copy()
    log('start', df)

    # Mid-only, drop locked/crossed
    mid = (df['best_bid_price'] + df['best_ask_price']) / 2.0
    rel_spread = (df['best_ask_price'] - df['best_bid_price']) / mid.replace(0.0, np.nan)
    df = df[(df['best_bid_price'] < df['best_ask_price']) & (rel_spread <= micro.spread_max)].copy()
    df['rel_spread'] = rel_spread.loc[df.index].values
    log('locked/crossed * spread', df)

    # Stale quotes (per instrument)
    if 'instrument_name' in df.columns:
        tmp = df.sort_values(['instrument_name','timestamp']).copy()
        dt = tmp.groupby('instrument_name')['timestamp'].diff().dt.total_seconds()
        tmp['stale_s'] = dt
        mask = tmp['stale_s'].isna() | (tmp['stale_s'] <= micro.stale_max_s)
        df = tmp.loc[mask].copy()
        log('stale', df)

    # Depth filter (adaptive per τ-bucket)
    if {'best_bid_amount','best_ask_amount'}.issubset(df.columns):
        tau_bin = pd.qcut(df['tau'], q=max(2, micro.n_tau_buckets), duplicates='drop')
        q_bid = df.groupby(tau_bin)['best_bid_amount'].transform(lambda s: s.quantile(micro.depth_q))
        q_ask = df.groupby(tau_bin)['best_ask_amount'].transform(lambda s: s.quantile(micro.depth_q))
        df = df[(df['best_bid_amount'] >= q_bid) & (df['best_ask_amount'] >= q_ask)].copy()
        log('depth', df)

    # τ guards
    tau_min_y = micro.tau_min_minutes / (365.0*24*60)
    tau_max_y = micro.tau_max_days / 365.0 if micro.tau_max_days is not None else np.inf
    df = df[(df['tau'] >= tau_min_y) & (df['tau'] <= tau_max_y)].copy()
    log('tauguards', df)

    # IV/vega for vega & MAD filters
    τ = df['tau'].to_numpy(); m = df['m'].to_numpy(); c = df['c_norm'].to_numpy()
    iv = implied_vol_from_c_norm(c, m, τ)             # [`utils.black_scholes.implied_vol_from_c_norm`](utils/black_scholes.py)
    vega = bs_vega_norm(m, τ, iv)                     # [`utils.black_scholes.bs_vega_norm`](utils/black_scholes.py)
    df['iv'] = iv; df['vega'] = vega
    

    # Vega guard (drop ultra ITM/OTM) by τ-bucket percentile
    tau_bin = pd.qcut(df['tau'], q=max(2, micro.n_tau_buckets), duplicates='drop')
    # new (keeps identical index):
    q = float(micro.vega_min_pct)  # e.g., 0.15 for 15th percentile
    vega_thr = df.groupby(tau_bin)['vega'].transform('quantile', q=q)
    df = df[df['vega'] >= vega_thr].copy()
    log('vega_guard', df)

    # MAD outlier trim on IV within (τ_bin, m_bin)
    m_bin = pd.qcut(df['m'], q=max(2, micro.n_m_buckets), duplicates='drop')
    grp = df.groupby([tau_bin, m_bin])
    med = grp['iv'].transform('median')
    mad = grp['iv'].transform(lambda s: np.median(np.abs(s - np.median(s))) + 1e-9)
    z = np.abs(df['iv'] - med) / mad
    df = df[z <= micro.mad_k].copy()
    log('MAD', df)

    # Calendar/vertical sanity checks (heuristics, drop wider-spread offender)
    if micro.vertical_check or micro.calendar_check:
        keep = np.ones(len(df), dtype=bool)
        idx_arr = df.index.to_numpy()

        # Per timestamp
        for ts, idxs in df.groupby('timestamp').groups.items():
            ids = np.asarray(list(idxs))

            if micro.vertical_check:
                for τv, row_ids in df.loc[ids].groupby('tau').groups.items():
                    rid = np.asarray(list(row_ids))
                    r = df.loc[rid].sort_values('m')
                    x = np.exp(r['m'].to_numpy()); c = r['c_norm'].to_numpy()
                    rs = r['rel_spread'].to_numpy(); id_sorted = r.index.to_numpy()

                    # monotone decreasing in m
                    bad = np.where(np.diff(c) > 1e-12)[0]
                    for j in bad:
                        drop_id = id_sorted[j if rs[j] >= rs[j+1] else j+1]
                        keep[np.where(idx_arr == drop_id)[0][0]] = False

                    # convex in K
                    if len(x) >= 3:
                        sL = (c[1:] - c[:-1]) / np.maximum(x[1:] - x[:-1], 1e-12)
                        sR = sL[1:]
                        sL = sL[:-1]
                        neg = np.where(sR - sL < -1e-12)[0]   # middle of (j-1, j, j+1)
                        for k in neg:
                            trip = np.array([k, k+1, k+2])
                            pick = trip[np.argmax(rs[trip])]
                            drop_id = id_sorted[pick]
                            keep[np.where(idx_arr == drop_id)[0][0]] = False

            if micro.calendar_check:
                r2 = df.loc[ids].copy()
                r2['τb'] = pd.qcut(r2['tau'], q=max(2, micro.n_tau_buckets), duplicates='drop')
                r2['mb'] = pd.qcut(r2['m'],   q=max(2, micro.n_m_buckets),   duplicates='drop')
                for _, g in r2.groupby('mb'):
                    if g.shape[0] < 2: 
                        continue
                    g = g.sort_values('tau')
                    c = g['c_norm'].to_numpy(); rs = g['rel_spread'].to_numpy()
                    viol = np.where(np.diff(c) < -1e-12)[0]  # longer τ < shorter τ
                    for j in viol:
                        drop_id = g.index[j if rs[j] >= rs[j+1] else j+1]
                        keep[np.where(idx_arr == drop_id)[0][0]] = False

        df = df[keep]

    return df.drop(columns=[c for c in ['iv','vega','rel_spread','stale_s'] if c in df.columns])

# Reading in jsonl file
def read_clean_jsonl(file_path):
    file = pd.read_json(file_path, lines = True)

    # Convert timestamp column to datetime
    file['timestamp'] = pd.to_datetime(
    file['timestamp'],
    format='%Y-%m-%d %H:%M:%S.%f'  # speeds up parsing, optional
    )

    stats_df = json_normalize(file['stats']).add_prefix('stats_')

    file = file.join(stats_df).drop(columns = ['stats'])

    return file
