import pandas as pd

class Indicator():
    '''
    This is an example Indicator class
    An Indicator should append values to the passed in dataframe as new columns
    any params used should be defined and stored in the class __init__ method
    this way the Indicator can be called multiple times with different params
    '''
    def __init__(self):
        pass
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
class MetaIndicator(Indicator):
    '''
    This is an example MetaIndicator class
    A MetaIndicator operates on underlying indicator(s)
    Useful for things like taking the difference between two indicators, or making a nonstationary indicator stationary
    '''
    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        kwargs : any parameters to be passed to the modifier
        '''
        pass
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        calculates the indicator and returns a new dataframe of only the indicator values
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with the OHLC data to be used for indicator calculations

        Returns
        -------
        indicator_df : pandas.DataFrame
            Dataframe of the Indicator values
        '''
        raise NotImplementedError
    
    
    def get_default_label(self):
        '''
        returns the class name and any set params, and should be preferred as the column name of the 
        Indicator_df if only a single Indicator is returned
        params set to none or starting with _ are not included in this name
        '''
        params = []
        for key, value in self.__dict__.items():
            if isinstance(value, Indicator):
                params.append(f'{key}={value.get_default_label()}')
            elif (not key.startswith('_') and not value is None):
                params.append(f'{key}={value}')
        param_str = f'({",".join(params)})'
        return self.__class__.__name__ + param_str