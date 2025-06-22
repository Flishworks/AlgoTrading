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

def infer_interval(df, monte_carlo_n=100):
    '''
    infers an interval from the index of the dataframe. 
    a monte_carlo approach is used to handle weekends/holidays
    returns a string using the following formatting conventions 
    compatible with openbb:
        1m = One Minute
        1h = One Hour
        1d = One Day
        1W = One Week
        1M = One Month
    '''
    intervals = []
    for i in range(monte_carlo_n):
        # monte carlo approach to handl epotential weekends/holidays
        rand_idx = np.random.randint(0, len(df)-1)
        intervals.append(df.index[rand_idx+1] - df.index[rand_idx])
    intervals = pd.Series(intervals)
    interval = intervals.mode()[0]
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
    
def detect_nonstationary_columns(df, threshold=0.05):
    '''
    Detects non-stationary columns in a dataframe using the Augmented Dickey-Fuller test.
    Returns a list of non-stationary columns.
    '''
    from statsmodels.tsa.stattools import adfuller
    
    nonstationary_columns = []
    
    for column in df.columns:
        try:
            result = adfuller(df[column])
            p_value = result[1]
            
            if p_value > threshold:
                nonstationary_columns.append(column)
        
        except Exception as e:
            print(f"Error processing column {column}: {e}")
    
    return nonstationary_columns


def diff_stationarizer(df, columns=None, lag=1):
    '''
    Differencing function to make a dataframe stationary.
    '''
    if columns is None:
        columns = df.columns
    
    for column in columns:
        df[column] = df[column].diff(periods=lag).dropna()
    
    return df.dropna()
    
    
def make_stationary(df, methods, nonstationary_columns=None):
    '''
    Makes the dataframe stationary by differencing the non-stationary columns.
    '''
    if nonstationary_columns is None:
        nonstationary_columns = detect_nonstationary_columns(df)
    for method in methods:
        if method == 'diff':
            for column in nonstationary_columns:
                df[column] = df[column].diff().dropna()
        elif method == 'log':
            for column in nonstationary_columns:
                df[column] = np.log(df[column]).diff().dropna()
        elif method == 'sqrt':
            for column in nonstationary_columns:
                df[column] = np.sqrt(df[column]).diff().dropna()
        elif method == 'boxcox':
            from scipy import stats
            for column in nonstationary_columns:
                df[column], _ = stats.boxcox(df[column])
                df[column] = df[column].diff().dropna()
    
    return df.dropna()