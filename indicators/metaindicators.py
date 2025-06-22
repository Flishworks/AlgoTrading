import pandas as pd
import numpy as np
from indicators.proto import Indicator, MetaIndicator
 
class diff(MetaIndicator):
    '''
    Diffs a single indicator by a period
    '''
    def __init__(self, indicator: Indicator, period: int = 1):
        self.ind = indicator
        self.period = period
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = self.ind.calculate(df).diff(periods=self.period).values
        return indicator_df
    
class subtract(MetaIndicator):
    '''
    Difference between two indicators (ind_1  - ind_2)
    '''
    def __init__(self, ind_1: Indicator, ind_2: Indicator):
        self.ind_1 = ind_1
        self.ind_2 = ind_2
        
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = (self.ind_1.calculate(df).values - self.ind_2.calculate(df).values)
        return indicator_df
    
class divide(MetaIndicator):
    '''
    Divides an indicator by a second indicator (ind_1 / ind_2)
    '''
    def __init__(self, ind_1: Indicator, ind_2: Indicator):
        self.ind_1 = ind_1
        self.ind_2 = ind_2
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        indicator_df = pd.DataFrame(index=df.index)
        result = (self.ind_1.calculate(df).values / self.ind_2.calculate(df).values)
        indicator_df[self.get_default_label()] = result
        return indicator_df
    
class rolling_znorm(MetaIndicator):
    '''
    Rolling z-score normalization of an indicator
    '''
    def __init__(self, ind: Indicator, period: int = 50):
        self.ind = ind
        self.period = period
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        indicator_df = pd.DataFrame(index=df.index)
        ind_values = self.ind.calculate(df)
        indicator_df[self.get_default_label()] = (ind_values - ind_values.rolling(self.period).mean()) / ind_values.rolling(self.period).std()
        #fill nans and infs
        indicator_df[self.get_default_label()].fillna(0, inplace=True)
        indicator_df[self.get_default_label()].replace([np.inf, -np.inf], 0, inplace=True)
        return indicator_df
    
class EMA(MetaIndicator):
    '''
    Exponential moving average of an indicator
    '''
    def __init__(self, ind: Indicator, period: int = 14):
        self.ind = ind
        self.period = period
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        indicator_df = pd.DataFrame(index=df.index)
        ind_values = self.ind.calculate(df)
        indicator_df[self.get_default_label()] = ind_values.ewm(span=self.period).mean()
        return indicator_df
    

class SMA(MetaIndicator):
    '''
    Simple moving average of an indicator
    '''
    def __init__(self, ind: Indicator, period: int = 14):
        self.ind = ind
        self.period = period
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        indicator_df = pd.DataFrame(index=df.index)
        ind_values = self.ind.calculate(df)
        indicator_df[self.get_default_label()] = ind_values.rolling(self.period).mean()
        return indicator_df