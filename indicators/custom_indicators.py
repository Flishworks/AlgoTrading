import pandas as pd
import numpy as np
from openbb import obb
import utils.ohlc_utils as ut
from utils import local_data_interface as ldi
from indicators.proto import Indicator
from scipy import signal

class ticker(Indicator):
    '''
    Simply copys a key from the dataframe. Useful for building MetaIndicators.
    '''
    def __init__(self, key = 'close'):
        self.key = key
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = df[self.key]
        return indicator_df
class SMA(Indicator):
    '''
    Simple Moving Average
    '''
    def __init__(self, period = 20, key='close'):
        self.period = period
        self.key = key
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = df[self.key].rolling(self.period).mean()
        return indicator_df
    
class EMA(Indicator):
    '''
    Exponential Moving Average
    '''
    def __init__(self, period = 20, key='close'):
        self.period = period
        self.key = key
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = df[self.key].ewm(span=self.period, adjust=False).mean()
        return indicator_df

class iirFilt(Indicator):
    """
    Applies an IIR filter to the input data. The filter is defined by the parameters passed to the constructor.
    lc: low cutoff frequency, in relative terms (0.0 to 1.0)
    hc: high cutoff frequency, in relative terms (0.0 to 1.0)
    If both lc and hc are provided, a bandpass filter is applied. Otherwise, a highpass or lowpass filter is applied.
    """
    def __init__(self, lc = None, hc = None, order=3, key='close'):
        self.lc = lc
        self.hc = hc
        self.order = order
        self.key = key

    def calculate(self, df):
        sig = df[self.key].values
        sos = None
        if self.lc is not None and self.hc is not None:
            sos = signal.butter(self.order, [self.lc, self.hc], btype='bandpass', output="sos")
        elif self.lc is not None:
            sos = signal.butter(self.order, self.lc, btype='highpass', output="sos")
        elif self.hc is not None:
            sos = signal.butter(self.order, self.hc, btype='lowpass', output="sos")
        
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = signal.sosfiltfilt(sos, sig)
        return indicator_df
    
class tears_bottom(Indicator):
    '''
    Tears Bottom
    '''
    def __init__(self, ema = 21):
        self.ema = ema
        
    def calculate(self, df):
        current_close_price = df['close']
        current_open_price = df['open']
        last_close_price = df['close'].shift(1)
        last_open_price = df['open'].shift(1)
        ema = df['close'].ewm(span=self.ema, adjust=False).mean()
        
        output = np.zeros(len(df))
        for i in range(1, len(df)):
            if last_close_price[i] - ema[i] < 0: # last candle below ema
                if last_close_price[i] < last_open_price[i]: # last candle was red
                    if current_close_price[i] > current_open_price[i]: # current candle green
                        if current_open_price[i] < last_close_price[i]: # opened below yesterdays close
                            output[i] = 1
        return pd.DataFrame({self.get_default_label() : output}, index=df.index)
    

class normalized_relative_price(Indicator):
    '''
    Normalized Relative Price
    Calculates the ratio of the stock price to the SP500 price, both normalized by the moving average
    '''
    def __init__(self, relative_to = 'spy', period = 30, key='close'):
        self.relative_to = relative_to
        self.period = period
        self.key = key
    
    def calculate(self, df):
        interval = ut.infer_interval(df)
        relative_price = ldi.get_ticker(symbol=self.relative_to, start_date=df.index[0], end_date=df.index[-1], interval=interval)[self.key]
        relative_normalized = (relative_price - relative_price.rolling(self.period).mean()) / relative_price.rolling(self.period).std() + 10 # offset to avoid division by zero
        current_stock_price = df[self.key]
        current_normalized_stock_price = (current_stock_price - current_stock_price.rolling(self.period).mean()) / current_stock_price.rolling(self.period).std() + 10 # offset to avoid division by zero
        price_ratio = current_normalized_stock_price / relative_normalized
        return pd.DataFrame({self.get_default_label() : price_ratio}, index=df.index)
    

class percent_from_local_high(Indicator):
    '''
    Calculates the percentage from the local high over a period
    '''
    def __init__(self, period=100, key='close'):
        self.period = period
        self.key = key
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        local_max = df['high'].rolling(self.period).max()
        indicator_df[self.get_default_label()] = df[self.key] / local_max
        return indicator_df
    
class percent_from_local_low(Indicator):
    '''
    Calculates the percentage from the local minimum over a period
    '''
    def __init__(self, period=100, key='close'):
        self.period = period
        self.key = key
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        local_min = df['low'].rolling(self.period).min()
        indicator_df[self.get_default_label()] = df[self.key] / local_min - 1
        return indicator_df

