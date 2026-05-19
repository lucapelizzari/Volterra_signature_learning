import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)





def get_var_dataset(window_size, batch_size=5000, dim=3, phi=0.8, sigma=0.5):
    def multi_AR(window_size, dim=3, phi=0.8, sigma=0.5, burn_in=200):
        window_size = window_size + burn_in
        xt = np.zeros((window_size, dim))
        one = np.ones(dim)
        ide = np.identity(dim)
        MU = np.zeros(dim)
        COV = sigma * one + (1 - sigma) * ide
        W = np.random.multivariate_normal(MU, COV, window_size)
        for i in range(dim):
            xt[0, i] = 0
        for t in range(window_size - 1):
            xt[t + 1] = phi * xt[t] + W[t]
        return xt[burn_in:]

    var_samples = []
    for i in range(batch_size):
        tmp = multi_AR(window_size, dim, phi=phi, sigma=sigma)
        var_samples.append(tmp)
    data_raw = torch.from_numpy(np.array(var_samples)).float()

    def get_pipeline():
        transforms = list()
        transforms.append(('standard_scale', StandardScalerTS(axis=(0, 1))))  # standard scale
        pipeline = Pipeline(steps=transforms)
        return pipeline

    pipeline = get_pipeline()
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed


def get_arch_dataset(window_size, lag=4, bt=0.055, N=5000, dim=1):
    """
    Creates the dataset: loads data.

    :param data_path: :param t_lag: :param device: :return:
    """

    def get_raw_data(N=5000, lag=4, T=2000, omega=0.00001, bt=0.055, burn_in=2000):
        beta = bt * np.ones(lag)
        eps = np.random.randn(N, T + burn_in)
        logrtn = np.zeros((N, T + burn_in))

        initial_arch = omega / (1 - beta[0])

        arch = initial_arch + np.zeros((N, T + burn_in))

        logrtn[:, :lag] = np.sqrt(arch[:, :lag]) * eps[:, :lag]

        for t in range(lag - 1, T + burn_in - 1):
            arch[:, t + 1] = omega + np.matmul(beta.reshape(1, -1), np.square(
                logrtn[:, t - lag + 1:t + 1]).transpose())  # * (logrtn[:, t] < 0.)
            logrtn[:, t + 1] = np.sqrt(arch[:, t + 1]) * eps[:, t + 1]
        return arch[:, burn_in:], logrtn[:, burn_in:]

    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    _, logrtn = get_raw_data(T=window_size, N=N, bt=bt)
    data_raw = torch.from_numpy(logrtn[..., None]).float()
    data_pre = pipeline.transform(data_raw)
    return pipeline, data_raw, data_pre


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def rolling_window(x, x_lag, add_batch_dim=True):
    if add_batch_dim:
        x = x[None, ...]
    return torch.cat([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], dim=0)




def get_equities_dataset(assets=('SPX', 'DJI'), with_vol=True):
    """
    Get different returns series.
    """
    oxford = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
    print(oxford.shape)
    start = '2000-01-01 00:00:00+01:00'
    end = '2022-01-01 00:00:00+01:00'

    if assets == ('SPX',):
        df_asset = oxford[oxford['Symbol'] == '.SPX'].set_index('Unnamed: 0')

        price = np.log(df_asset[['close_price']].values)          # (T,1)
        logRV = np.log(df_asset[['medrv']].values)                # (T,1)
        vol = np.exp(0.5*logRV)
    
        price = price.reshape(1, -1, 1)                           # (1,T,1)
        logRV  = logRV.reshape(1, -1, 1)                          # (1,T,1)
        vol = vol.reshape(1, -1, 1)     
        data_raw = np.concatenate([price, vol], axis=-1)        # (1,T,2)
    elif assets == ('SPX', 'DJI'):
        df_spx = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])[start:end]
        df_dji = oxford[oxford['Symbol'] == '.DJI'].set_index(['Unnamed: 0'])[start:end]
        index = df_dji.index.intersection(df_spx.index)
        df_dji = df_dji.loc[index]
        df_spx = df_spx.loc[index]
        price_spx = np.log(df_spx[['close_price']].values)
        rtn_spx = (price_spx[1:] - price_spx[:-1]).reshape(1, -1, 1)
        vol_spx = np.log(df_spx[['medrv']].values).reshape(1, -1, 1)
        price_dji = np.log(df_dji[['close_price']].values)
        rtn_dji = (price_dji[1:] - price_dji[:-1]).reshape(1, -1, 1)
        vol_dji = np.log(df_dji[['medrv']].values).reshape(1, -1, 1)
        #data_raw = np.concatenate([rtn_spx, vol_spx[:, 1:], rtn_dji, vol_dji[:, 1:]], axis=-1)
        data_raw = np.concatenate([price_spx, vol_spx[:, 1:], price_dji, vol_dji[:, 1:]], axis=-1)
    else:
        raise NotImplementedError()
    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed
import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline as SklearnPipeline

# you already have this in your project
# from your_module import StandardScalerTS


def get_spx_vix_dataset(
    assets=("SPX",),
    with_vol=True,
    start="2000-01-01",
    end="2026-01-01",
):
    """
    Returns:
      pipeline, data_raw, data_preprocessed

    data_raw shape conventions (same as your code):
      - single asset: (1, T, d)
      - multi asset:  (1, T, d_multi)

    Channels (when assets=('SPX',) and with_vol=True):
      [log_price_spx, vix]   # vix in levels (percent units like 15,20,...)

    Notes:
      - SPX from Stooq '^spx' (index close)
      - VIX from FRED 'VIXCLS'
      - We inner-join dates and forward-fill VIX (holidays)
    """

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # -------------------------
    # Load SPX from Stooq (daily)
    # -------------------------
    # Stooq returns columns: Date, Open, High, Low, Close, Volume
    # We only need Close.
    spx_url = "https://stooq.com/q/d/l/?s=^spx&i=d"
    spx = pd.read_csv(spx_url)
    spx["Date"] = pd.to_datetime(spx["Date"])
    spx = spx.set_index("Date").sort_index()
    spx = spx.loc[start_dt:end_dt]
    spx = spx.rename(columns={"Close": "spx_close"})[["spx_close"]]

    # -------------------------
    # Load VIX from FRED (daily close)
    # -------------------------
    vix_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    vix = pd.read_csv(vix_url)
    vix["observation_date"] = pd.to_datetime(vix["observation_date"])
    vix = vix.set_index("observation_date").sort_index()
    vix = vix.rename(columns={"VIXCLS": "vix"})
    # FRED uses "." for missing sometimes
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    vix = vix.loc[start_dt:end_dt]

    # -------------------------
    # Align
    # -------------------------
    df = spx.join(vix, how="inner")
    # If you prefer keep SPX calendar and fill VIX:
    # df = spx.join(vix, how="left"); df["vix"] = df["vix"].ffill()

    df = df.dropna(subset=["spx_close", "vix"])

    # build features
    log_price_spx = np.log(df[["spx_close"]].values)  # (T,1)
    vix_level = df[["vix"]].values                    # (T,1)

    # shape (1,T,2)
    if assets == ("SPX",):
        if with_vol:
            data_raw = np.concatenate([log_price_spx, vix_level], axis=-1)
        else:
            data_raw = log_price_spx
        data_raw = data_raw.reshape(1, -1, data_raw.shape[-1])

    elif assets == ("SPX", "VIX"):
        # explicit two-“asset” style: still return as channels
        data_raw = np.concatenate([log_price_spx, vix_level], axis=-1).reshape(1, -1, 2)

    else:
        raise NotImplementedError("This helper currently supports assets=('SPX',) or ('SPX','VIX').")

    data_raw = torch.from_numpy(data_raw).float()

    pipeline = Pipeline(steps=[("standard_scale", StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)

    return pipeline, data_raw, data_preprocessed
    
def download_man_ahl_dataset():
    import requests
    from zipfile import ZipFile
    url = 'https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip'
    r = requests.get(url)
    with open('./oxford.zip', 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=100 * 1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile('./oxford.zip')
    zf.extractall(path='./data')
    zf.close()
    os.remove('./oxford.zip')

import time
import numpy as np
import torch
import ccxt

# assumes you have these in your project like for equities
# from sklearn.pipeline import Pipeline
# from your_module import StandardScalerTS

import time
import numpy as np
import torch
import ccxt

def _fetch_ohlcv_all(exchange, symbol, timeframe, since_ms, until_ms=None, limit=1000):
    """Paginate ccxt OHLCV: returns np.array shape (N, 6) [ts, o,h,l,c,v]."""
    all_rows = []
    step_ms = exchange.parse_timeframe(timeframe) * 1000

    while True:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not rows:
            break

        # If until_ms is set, keep only rows strictly before until_ms
        if until_ms is not None:
            rows = [r for r in rows if r[0] < until_ms]
            if not rows:
                break

        all_rows.extend(rows)

        last_ts = rows[-1][0]
        next_since = last_ts + step_ms
        if until_ms is not None and next_since >= until_ms:
            break
        if next_since <= since_ms:  # safety
            break
        since_ms = next_since

        if exchange.enableRateLimit:
            time.sleep(exchange.rateLimit / 1000)

    return np.asarray(all_rows, dtype=np.float64)

def get_crypto_dataset(
    assets=("BTC", "ETH"),
    quote="USDT",
    timeframe="1h",
    start_iso="2023-01-01T00:00:00Z",
    end_iso="2025-01-01T00:00:00Z",
    with_vol=False,
    exchange_id="binance",
):
    """
    Returns:
      pipeline, data_raw, data_preprocessed
    data_raw / data_preprocessed: torch tensors (1, T, d)
    """
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    since = ex.parse8601(start_iso)
    until = ex.parse8601(end_iso)

    feats = []
    for a in assets:
        sym = f"{a}/{quote}"
        rows = _fetch_ohlcv_all(ex, sym, timeframe=timeframe, since_ms=since, until_ms=until)

        if rows.shape[0] < 3:
            raise RuntimeError(f"Not enough data for {sym} ({rows.shape[0]} rows).")

        close = rows[:, 4]
        vol   = rows[:, 5]

        logp = np.log(close + 1e-12).reshape(1, -1, 1)
        #rtn = (logp[1:] - logp[:-1]).reshape(1, -1, 1)  # (1, T, 1)
        feats.append(logp)

        if with_vol:
            lv = np.log(vol[1:] + 1e-12).reshape(1, -1, 1)
            feats.append(lv)

    # align lengths across assets (take common min T)
    T = min(f.shape[1] for f in feats)
    feats = [f[:, -T:, :] for f in feats]

    data_raw = np.concatenate(feats, axis=-1)            # (1, T, d)
    data_raw = torch.from_numpy(data_raw).float()

    pipeline = Pipeline(steps=[("standard_scale", StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    

    return pipeline, data_raw, data_preprocessed


import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def get_electricity_dataset(
    url="https://raw.githubusercontent.com/SasinduChanakaPiyumal/Electricity_Data/refs/heads/main/electricity.csv",
    outputs=('ForecastWindProduction','SystemLoadEA','SMPEA','ActualWindProduction','SystemLoadEP2','SMPEP2'),
    scale="minmax",                 # "minmax" or "standard" (minmax matches your snippet)
    return_index=False,
):
    """
    Loads and cleans the electricity dataset, returning:
      pipeline, data_raw, data_preprocessed
    where
      data_raw: torch.FloatTensor (1, T, d)
      data_preprocessed: torch.FloatTensor (1, T, d)
    """
    df = pd.read_csv(
        url,
        index_col=0,
        parse_dates=[0],
        na_values=["?", "", "NA", "N/A", None]
    )

    # ensure numeric columns (robust)
    cols_to_numeric = [
        'ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 'ORKWindspeed',
        'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2'
    ]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # drop rows missing essential weather (as in your snippet)
    df = df.dropna(subset=['ORKTemperature','ORKWindspeed'])

    # filter price range (your condition was always True; corrected to AND)
    if 'SMPEP2' in df.columns:
        df = df[(df['SMPEP2'] > 0) & (df['SMPEP2'] <= 550)]

    # fill rules
    fill_with_median = ['ForecastWindProduction','SystemLoadEA','SMPEA','ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
    for col in fill_with_median:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if 'CO2Intensity' in df.columns:
        df['CO2Intensity'] = df['CO2Intensity'].fillna(df['CO2Intensity'].mean())

    # select outputs
    outputs = list(outputs)
    missing = [c for c in outputs if c not in df.columns]
    if missing:
        raise ValueError(f"Requested outputs missing in df: {missing}")

    X = df[outputs].to_numpy(dtype=float)  # (T,d)

    # scale
    if scale == "minmax":
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        pipeline = scaler
    elif scale == "standard":
        # simple standardization without custom TS scaler
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        Xs = (X - mu) / sd
        pipeline = ("standard", mu, sd)
    elif scale == "TS":
        pipeline = Pipeline(steps=[("standard_scale", StandardScalerTS(axis=(0, 1)))])
        Xs = pipeline.transform(X)
    else:
        raise ValueError("scale must be 'minmax' or 'standard'")

    data_raw = torch.from_numpy(X[None, :, :]).float()   # (1,T,d)
    data_preprocessed = torch.from_numpy(Xs[None, :, :]).float()

    if return_index:
        return pipeline, data_raw, data_preprocessed, df.index
    return pipeline, data_raw, data_preprocessed

def get_data(data_type, p, q, **data_params):
    if data_type == 'VAR':
        pipeline, x_real_raw, x_real = get_var_dataset(40000, batch_size=1, **data_params)

    elif data_type == 'STOCKS':
        pipeline, x_real_raw, x_real = get_equities_dataset(**data_params)
    elif data_type == 'SPXVIX':
        pipeline, x_real_raw, x_real = get_spx_vix_dataset(**data_params)

    elif data_type == 'CRYPTO':
        pipeline, x_real_raw, x_real = get_crypto_dataset(**data_params)

    elif data_type == 'ARCH':
        pipeline, x_real_raw, x_real = get_arch_dataset(40000, N=1, **data_params)

    elif data_type == 'ECG':
        pipeline, x_real_raw, x_real = get_mit_arrythmia_dataset(**data_params)

    elif data_type == 'ELECTRICITY':
        pipeline, x_real_raw, x_real = get_electricity_dataset(**data_params)

    elif data_type == 'ELECTRICITY_OPSD':
        pipeline, x_real_raw, x_real = get_opsd_electricity_dataset(**data_params)

    elif data_type == 'OVX':
        pipeline, x_real_raw, x_real = get_ovx_dataset(**data_params)

    # optional: a generic FRED mode if you want many series easily
    elif data_type == 'FRED':
        pipeline, x_real_raw, x_real = get_fred_dataset(**data_params)

    else:
        raise NotImplementedError(f'Dataset {data_type} not valid')

    assert x_real.shape[0] == 1
    return pipeline, x_real_raw, x_real[0]


import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def get_fred_dataset(
    series_ids=("OVXCLS",),     # e.g. ("OVXCLS",) or ("OVXCLS","VIXCLS")
    outputs=None,              # subset of series_ids; default = series_ids
    start=None,                # e.g. "2010-01-01"
    end=None,                  # e.g. "2024-12-31"
    scale="standard",          # "minmax" or "standard"
    transform=None,            # None, "log", or "log1p"
    fill="ffill",              # "drop", "ffill", "interpolate"
    return_index=False,
):
    """
    Loads one or multiple FRED time series and returns:
      pipeline, data_raw, data_preprocessed
    where
      data_raw: torch.FloatTensor (1, T, d)
      data_preprocessed: torch.FloatTensor (1, T, d)

    Notes:
      - FRED series often have missing days (weekends/holidays). With fill="ffill",
        you keep business-day spacing; if you later need a strict daily grid,
        reindex outside this function.
      - This function fits the scaler on the full series (like your electricity loader).
        If you want no leakage, split train/test first and fit scaler on train only.
    """

    def _fetch_one(sid: str) -> pd.DataFrame:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        df = pd.read_csv(
            url,
            parse_dates=["observation_date"],
            na_values=[".", "NA", "N/A", "", None],
        )
        if sid not in df.columns:
            # FRED returns "DATE,<sid>" typically; if not, handle gracefully
            # by taking the second column name.
            val_col = [c for c in df.columns if c != "observation_date"][0]
            df = df.rename(columns={val_col: sid})
        df = df[["observation_date", sid]].set_index("observation_date").sort_index()
        df[sid] = pd.to_numeric(df[sid], errors="coerce")
        return df

    # ---- download & merge
    series_ids = tuple(series_ids)
    dfs = [_fetch_one(sid) for sid in series_ids]
    df = pd.concat(dfs, axis=1, join="outer").sort_index()

    # ---- optional date filter
    if start is not None:
        df = df.loc[pd.to_datetime(start):]
    if end is not None:
        df = df.loc[:pd.to_datetime(end)]

    # ---- choose outputs
    if outputs is None:
        outputs = list(series_ids)
    else:
        outputs = list(outputs)

    missing = [c for c in outputs if c not in df.columns]
    if missing:
        raise ValueError(f"Requested outputs missing in df: {missing}")

    # ---- handle missing values
    if fill == "drop":
        df = df.dropna(subset=outputs)
    elif fill == "ffill":
        df[outputs] = df[outputs].ffill().bfill()
        df = df.dropna(subset=outputs)
    elif fill == "interpolate":
        df[outputs] = df[outputs].interpolate(method="time").ffill().bfill()
        df = df.dropna(subset=outputs)
    else:
        raise ValueError("fill must be one of: 'drop', 'ffill', 'interpolate'")

    # ---- extract array
    X = df[outputs].to_numpy(dtype=float)  # (T, d)

    # ---- optional transform (often useful for vol indices)
    if transform is not None:
        if transform == "log":
            # OVX is positive; protect just in case
            X = np.log(np.clip(X, 1e-8, None))
        elif transform == "log1p":
            X = np.log1p(np.clip(X, 0.0, None))
        else:
            raise ValueError("transform must be one of: None, 'log', 'log1p'")

    # ---- scale
    if scale == "minmax":
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        pipeline = scaler
    elif scale == "standard":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        Xs = (X - mu) / sd
        pipeline = ("standard", mu, sd)
    else:
        raise ValueError("scale must be 'minmax' or 'standard'")

    data_raw = torch.from_numpy(X[None, :, :]).float()        # (1, T, d)
    data_preprocessed = torch.from_numpy(Xs[None, :, :]).float()

    if return_index:
        return pipeline, data_raw, data_preprocessed, df.index
    return pipeline, data_raw, data_preprocessed


def get_ovx_dataset(
    scale="standard",
    transform=None,          # consider "log" if you want “memory in volatility” to be clearer
    fill="ffill",
    return_index=False,
):
    """
    Convenience wrapper for OVX only.
    """
    return get_fred_dataset(
        series_ids=("OVXCLS",),
        outputs=("OVXCLS",),
        scale=scale,
        transform=transform,
        fill=fill,
        return_index=return_index,
    )

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def get_opsd_electricity_dataset(
    path="time_series_60min_singleindex.csv",
    country="AT",   # used only if outputs=None
    outputs=None,   # e.g. ("AT_price_day_ahead", "AT_load_actual_entsoe_transparency", ...)
    scale="minmax",                 # "minmax" or "standard"
    return_index=False,
    dropna_outputs=True,            # drop rows missing any chosen outputs
    fill_with_median=True,          # if dropna_outputs=False, fill NaNs with median
    price_col=None,                # optionally set (e.g. "AT_price_day_ahead") to filter out bad values
    price_min=None,                # e.g. 0.0
    price_max=None,                # e.g. 1000.0
):
    """
    Loads OPSD time_series_60min_singleindex.csv and returns:
      pipeline, data_raw, data_preprocessed
    where
      data_raw: torch.FloatTensor (1, T, d)
      data_preprocessed: torch.FloatTensor (1, T, d)

    Notes:
      - The OPSD file is wide (many columns); you should pass `outputs` explicitly.
      - If outputs=None, it auto-selects a small default set for the given `country`
        when available (price + load + wind + solar).
    """

    # ---- load
    df = pd.read_csv(path, na_values=["?", "", "NA", "N/A", None])

    # ---- timestamp handling
    ts_col = None
    for c in ["cet_cest_timestamp", "utc_timestamp", "timestamp", "date", "Datetime", "datetime"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(
            "Could not find a timestamp column. Expected one of: "
            "'cet_cest_timestamp', 'utc_timestamp', ... "
            f"Columns seen: {list(df.columns)[:20]}..."
        )

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()

    # ---- choose outputs
    if outputs is None:
        pref = f"{country}_"
        # try to pick a sensible multivariate set if present
        candidates = [
            f"{pref}price_day_ahead",
            f"{pref}load_actual_entsoe_transparency",
            f"{pref}wind_onshore_generation_actual",
            f"{pref}solar_generation_actual",
        ]
        outputs = [c for c in candidates if c in df.columns]
        if len(outputs) == 0:
            # fallback: pick the first few columns with the prefix
            pref_cols = [c for c in df.columns if c.startswith(pref)]
            if len(pref_cols) == 0:
                raise ValueError(f"No columns found with prefix '{pref}'.")
            outputs = pref_cols[:6]  # arbitrary small default
    else:
        outputs = list(outputs)

    missing = [c for c in outputs if c not in df.columns]
    if missing:
        raise ValueError(f"Requested outputs missing in df: {missing}")

    # ---- ensure numeric for chosen outputs
    for col in outputs:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- optional price filtering
    if price_col is not None:
        if price_col not in df.columns:
            raise ValueError(f"price_col='{price_col}' not found in df columns.")
        m = np.ones(len(df), dtype=bool)
        if price_min is not None:
            m &= df[price_col].to_numpy() >= price_min
        if price_max is not None:
            m &= df[price_col].to_numpy() <= price_max
        df = df.loc[m]

    # ---- missing value strategy
    if dropna_outputs:
        df = df.dropna(subset=outputs)
    else:
        if fill_with_median:
            for col in outputs:
                df[col] = df[col].fillna(df[col].median())
        else:
            df = df.dropna(subset=outputs)

    # ---- extract as (T, d)
    X = df[outputs].to_numpy(dtype=float)

    # ---- scale
    if scale == "minmax":
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        pipeline = scaler
    elif scale == "standard":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        Xs = (X - mu) / sd
        pipeline = ("standard", mu, sd)
    elif scale == "TS":
        # X is numpy (T, d)
        Xt = torch.from_numpy(X[None, :, :]).float()  # (1, T, d)
    
        pipeline = Pipeline(steps=[("standard_scale", StandardScalerTS(axis=(0, 1)))])
        Xs_t = pipeline.transform(Xt)                 # (1, T, d) torch
    
        data_raw = Xt
        data_preprocessed = Xs_t
    
        if return_index:
            return pipeline, data_raw, data_preprocessed, df.index
        return pipeline, data_raw, data_preprocessed
    else:
        raise ValueError("scale must be 'minmax' or 'standard'")

    data_raw = torch.from_numpy(X[None, :, :]).float()          # (1,T,d)
    data_preprocessed = torch.from_numpy(Xs[None, :, :]).float() # (1,T,d)

    if return_index:
        return pipeline, data_raw, data_preprocessed, df.index
    return pipeline, data_raw, data_preprocessed