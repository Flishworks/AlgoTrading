import pandas as pd
import pandas_ta as ta
from .custom_indicators import Indicator

# for indicator in ta.available_indicators():
#     exec(f'''  
#         class {indicator}(indicator):
#             def __init__(self, **kwargs):
#                 self.kwargs = kwargs
                
#             def calculate(self, df):
#                 return df.ta.{indicator}(**self.kwargs)
#     ''')
    
class all_candlestick_patterns(Indicator):
    '''
    All Candlestick Patterns
    '''
    def __init__(self, transform = None):
        self.transform = transform
        
    def calculate(self, df):
        if self.transform is not None:
            df = self.transform(df)
        return_df = df.ta.cdl_pattern(name='all')
        return return_df