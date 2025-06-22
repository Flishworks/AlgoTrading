import pandas as pd
import pandas_ta as ta
import numpy as np
class Extractor:
    def __init__(self, indicators: list):
        self.indicators = indicators
        

    def extract(self, df: pd.DataFrame, keep_original: bool = False) -> pd.DataFrame:
        '''
        Calculates indicators from the passed in dataframe and returns them in a new dataframe.
        If keep_original is True, the original dataframe columns are retained in the return DF. Otherwise they are omitted.
        '''
        og_df = df.copy()
        if keep_original:
            new_df = df.copy()
        else:
            new_df = pd.DataFrame(index=df.index)
        for indicator in self.indicators:
            try:
                indicator_df = indicator.calculate(og_df)
                #replace infs with np.nan
                indicator_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                new_df = pd.concat([new_df, indicator_df], axis=1)
            except Exception as e:
                print(f"Error with indicator {indicator.get_default_label()}: {e}")
        return new_df
