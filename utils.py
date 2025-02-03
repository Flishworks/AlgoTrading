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

def infer_interval(df):
    '''
    infers an interval from the index of the dataframe. 
    returns a string using the following formatting conventions 
    compatible with openbb:
        1m = One Minute
        1h = One Hour
        1d = One Day
        1W = One Week
        1M = One Month
    '''
    interval = df.index[-1] - df.index[-2]
    interval_minutes = interval.total_seconds() / 60
    if interval_minutes < 1:
        raise ValueError('Interval too small')
    if interval_minutes >= 1 and interval_minutes < 60:
        return str(round(interval_minutes)) + 'm'
    elif interval_minutes >= 60 and interval_minutes < 1440:
        return str(round(interval_minutes // 60)) + 'h'
    elif interval_minutes >= 1440 and interval_minutes < 10080:
        return str(round(interval_minutes // 1440)) + 'd'
    elif interval_minutes >= 10080 and interval_minutes < 40320:
        return str(round(interval_minutes // 10080)) + 'W'
    else:
        return str(round(interval_minutes // 40320)) + 'M'