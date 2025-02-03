import pandas as pd
import pandas_ta as ta

class Extractor:
    def __init__(self, indicators: list):
        self.indicators = indicators
        

    def extract(self, df: pd.DataFrame):
        new_df = df.copy()
        for indicator in self.indicators:
            new_df = pd.concat([new_df, indicator.calculate(df)], axis=1)
        return new_df
