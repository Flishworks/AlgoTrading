import pandas as pd
import numpy as np

def ohlc_resampler(df, freq='1H', round = True, fill='ffill'):
    times = pd.date_range(df.index[0], df.index[-1], freq=freq)
    if round:
        times = times.round(freq)
    _df = pd.DataFrame(index=times)
    
    agg_map = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    agg_dict = {}
    for df_key in df.columns:
        for agg_key in agg_map:
            if agg_key in df_key:
                agg_dict[df_key] = agg_map[agg_key]
    resampled = df.resample(freq).agg(agg_dict)
    _df = _df.join(resampled, how='left')
    _df.fillna(method=fill, inplace=True)
    return _df

def normalize_by_start(df):
    return df / df.iloc[0]