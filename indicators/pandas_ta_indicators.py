import pandas as pd
import pandas_ta as ta
from indicators.proto import Indicator
class all_candlestick_patterns(Indicator):
    '''
    All Candlestick Patterns
    '''
    def __init__(self, decay_period=None):
        self.decay_period = decay_period
        
    def calculate(self, df):
        result = df.ta.cdl_pattern(name='all')
        result /= 100 # Normalize to [-1, 1]
        if self.decay_period is not None:
            # make each positive or negative value decay to 0 linearly over a period of self.decay_period
            result = result.ewm(span=self.decay_period).mean()
            #append 'decayed' to the label
            result.columns = [f'{col}(decayed_{self.decay_period})' for col in result.columns]
        return result