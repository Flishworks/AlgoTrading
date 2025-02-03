import pandas as pd
import numpy as np
from openbb import obb
import utils as ut

class Indicator():
    '''
    This is an example Indicator class
    An Indicator should append values to the passed in dataframe as new columns
    any params used should be defined ans stored in the class __init__ method
    this way the Indicator can be called multiple times with different params
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        '''
          Parameters
            ----------
            df : pandas.DataFrame
                Dataframe with the OHLC data to be used for calculations

            Returns
            -------
            Indicator_df : pandas.DataFrame
                Dataframe of the Indicator values'''
        raise NotImplementedError
    
    def get_default_label(self):
        '''
        returns the class name and any set params, and should be preferred as the column name of the 
        Indicator_df if only a single Indicator is returned
        params set to none or starting with _ are not included in this name
        '''
        params = [f'{key}={value}' for key, value in self.__dict__.items() if (not key.startswith('_') and not value is None)]
        if len(params):
            param_str = f'({",".join(params)})'
        else:
            param_str = '()'
        return self.__class__.__name__ + param_str

class SMA(Indicator):
    '''
    Simple Moving Average
    '''
    def __init__(self, period = 20):
        self.period = period
        
    def calculate(self, df):
        df[self.get_default_label()] = df['close'].rolling(self.period).mean()
        return df
    
class EMA(Indicator):
    '''
    Exponential Moving Average
    '''
    def __init__(self, period = 20):
        self.period = period
        
    def calculate(self, df):
        df[self.get_default_label()] = df['close'].ewm(span=self.period, adjust=False).mean()
        return df
    
class Tears_Bottom(Indicator):
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
        return pd.DataFrame({self.get_default_label() : output})
    

class Normalized_Relative_Price(Indicator):
    '''
    Normalized Relative Price
    Calculates the ratio of the stock price to the SP500 price, both normalized by the moving average
    '''


    def __init__(self, relative_to = 'spy', length = 14, col='close'):
        self.relative_to = relative_to
        self.length = length
        self.col = col
    
    def calculate(self, df):
        interval = ut.infer_interval(df)
        sp500_price = obb.equity.price.historical(symbol=self.relative_to, provider="yfinance", start_date=df.index[0], end_date=df.index[-1], interval='1d').to_df()[self.col]
        sp500_normalized = sp500_price / sp500_price.rolling(self.length).mean()
        current_stock_price = df[self.col]
        current_normalized_stock_price = current_stock_price / current_stock_price.rolling(self.length).mean()
        price_ratio = current_normalized_stock_price / sp500_normalized
        return pd.DataFrame({self.get_default_label() : price_ratio})
    
