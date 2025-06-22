import pandas as pd
import tulipy as ti
from indicators.proto import Indicator
import numpy as np

class ad(Indicator):
    '''
    Accumulation/Distribution Line
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = ti.ad(df['high'].values, df['low'].values, df['close'].values, df['volume'].values.astype(float))
        return indicator_df
    
class adosc(Indicator):
    '''
    Accumulation/Distribution Oscillator
    '''
    def __init__(self, short = 3, long = 10):
        self.short = short
        self.long = long
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        results = ti.adosc(df['high'].values, df['low'].values, df['close'].values, df['volume'].values.astype(float), self.short, self.long)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(results)), results])
        return indicator_df

class adx(Indicator):
    '''
    Average Directional Movement Index
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        results = ti.adx(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(results)), results])
        return indicator_df
    
class adxr(Indicator):
    '''
    Average Directional Movement Index Rating
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        results = ti.adxr(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(results)), results])
        return indicator_df
    
class ao(Indicator):
    '''
    Awesome Oscillator
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        results = ti.ao(df['high'].values, df['low'].values)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(results)), results])
        return indicator_df
    
class apo(Indicator):
    '''
    Absolute Price Oscillator
    '''
    def __init__(self, short = 12, long = 26):
        self.short = short
        self.long = long
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.apo(df['close'].values, self.short, self.long)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class aroon_up(Indicator):
    '''
    Aroon up
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        aroon_down, aroon_up = ti.aroon(df['high'].values, df['low'].values, self.period)
        indicator_df[self.get_default_label() + "_up"] = np.concatenate([np.zeros(self.period), aroon_up])
        return indicator_df
    
class aroon_down(Indicator):
    '''
    Aroon down
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        aroon_down, aroon_up = ti.aroon(df['high'].values, df['low'].values, self.period)
        indicator_df[self.get_default_label() + "_down"] = np.concatenate([np.zeros(self.period), aroon_down])
        return indicator_df
    
class aroonosc(Indicator):
    '''
    Aroon Oscillator
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(self.period), ti.aroonosc(df['high'].values, df['low'].values, self.period)])
        return indicator_df
    
class atr(Indicator):
    '''
    Average True Range
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.atr(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class bbands_upper(Indicator):
    '''
    Bollinger Bands - upper band
    '''
    def __init__(self, key = 'close', period = 5, stddev = 2):
        self.key = key
        self.period = period
        self.stddev = stddev
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        upper, middle, lower = ti.bbands(df[self.key].values, self.period, self.stddev)
        pad_len = len(df) - len(upper)
        indicator_df[self.get_default_label() + "_upper"] = np.concatenate([np.zeros(pad_len), upper])
        return indicator_df

class bbands_middle(Indicator):
    '''
    Bollinger Bands - middle Band
    '''
    def __init__(self, key = 'close', period = 5, stddev = 2):
        self.key = key
        self.period = period
        self.stddev = stddev
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        upper, middle, lower = ti.bbands(df[self.key].values, self.period, self.stddev)
        pad_len = len(df) - len(middle)
        indicator_df[self.get_default_label() + "_middle"] = np.concatenate([np.zeros(pad_len), middle])
        return indicator_df
    
class bbands_lower(Indicator):
    '''
    Bollinger Bands - lower band
    '''
    def __init__(self, key = 'close', period = 5, stddev = 2):
        self.key = key
        self.period = period
        self.stddev = stddev
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        upper, middle, lower = ti.bbands(df[self.key].values, self.period, self.stddev)
        pad_len = len(df) - len(lower)
        indicator_df[self.get_default_label() + "_lower"] = np.concatenate([np.zeros(pad_len), lower])
        return indicator_df

class bop(Indicator):
    '''
    Balance of Power
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = ti.bop(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        return indicator_df
    
class cci(Indicator):
    '''
    Commodity Channel Index
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.cci(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class cmo(Indicator):
    '''
    Chande Momentum Oscillator
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.cmo(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class cvi(Indicator):
    '''
    Chaikin Volatility
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.cvi(df['high'].values, df['low'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df

class dema(Indicator):
    '''
    Double Exponential Moving Average
    '''
    def __init__(self, key='close', period = 30):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.dema(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class di_plus(Indicator):
    '''
    Directional Indicator
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        di_minus, di_plus = ti.di(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label() + "_plus"] = np.concatenate([np.zeros(len(df) - len(di_plus)), di_plus])
        return indicator_df

class di_minus(Indicator):
    '''
    Directional Indicator
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        di_minus, di_plus = ti.di(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label() + "_minus"] = np.concatenate([np.zeros(len(df) - len(di_minus)), di_minus])
        return indicator_df
    
class dm_plus(Indicator):
    '''
    Directional Movement
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        dm_minus, dm_plus = ti.dm(df['high'].values, df['low'].values, self.period)
        indicator_df[self.get_default_label() + "_plus"] = np.concatenate([np.zeros(len(df) - len(dm_plus)), dm_plus])
        return indicator_df   
    
class dm_minus(Indicator):
    '''
    Directional Movement
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        dm_minus, dm_plus = ti.dm(df['high'].values, df['low'].values, self.period)
        indicator_df[self.get_default_label() + "_minus"] = np.concatenate([np.zeros(len(df) - len(dm_minus)), dm_minus])
        return indicator_df   
    
class dpo(Indicator):
    '''
    Detrended Price Oscillator
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.dpo(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class dx(Indicator):
    '''
    Directional Movement Index
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.dx(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class emv(Indicator):
    '''
    Ease of Movement
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.emv(df['high'].values, df['low'].values, df['volume'].values.astype(float))
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class fisher(Indicator):
    '''
    Fisher Transform
    '''
    def __init__(self, period = 9):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        fisher, fisher_signal = ti.fisher(df['high'].values, df['low'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(fisher)), fisher])
        # df[self.get_default_label() + "_signal"] = np.concatenate([np.zeros(len(df) - len(fisher_signal)), fisher_signal]) # ignoreing signal line
        return indicator_df
    
class fosc(Indicator):
    '''
    Forecast Oscillator
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.fosc(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class kvo(Indicator):
    '''
    Klinger Volume Oscillator
    '''
    def __init__(self, short = 34, long = 55):
        self.short = short
        self.long = long
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        kvo = ti.kvo(df['high'].values, df['low'].values, df['close'].values, df['volume'].values.astype(float), self.short, self.long)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(kvo)), kvo])
        return indicator_df
    
class linregintercept(Indicator):
    '''
    Linear Regression Intercept
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.linregintercept(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df

class linregslope(Indicator):
    '''
    Linear Regression Slope
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.linregslope(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df

class macd(Indicator):
    '''
    Moving Average Convergence Divergence
    '''
    def __init__(self, key='close', short = 12, long = 26, signal = 9):
        self.key = key
        self.short = short
        self.long = long
        self.signal = signal
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        macd, macd_signal, macd_hist = ti.macd(df[self.key].values, self.short, self.long, self.signal)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(macd)), macd])
        return indicator_df
    
class macd_signal(Indicator):
    '''
    Moving Average Convergence Divergence
    '''
    def __init__(self, key='close', short = 12, long = 26, signal = 9):
        self.key = key
        self.short = short
        self.long = long
        self.signal = signal
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        macd, macd_signal, macd_hist = ti.macd(df[self.key].values, self.short, self.long, self.signal)
        indicator_df[self.get_default_label() + "_signal"] = np.concatenate([np.zeros(len(df) - len(macd_signal)), macd_signal])
        return indicator_df
    
class macd_hist(Indicator):
    '''
    Moving Average Convergence Divergence
    '''
    def __init__(self, key='close', short = 12, long = 26, signal = 9):
        self.key = key
        self.short = short
        self.long = long
        self.signal = signal
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        macd, macd_signal, macd_hist = ti.macd(df[self.key].values, self.short, self.long, self.signal)
        indicator_df[self.get_default_label() + "_hist"] = np.concatenate([np.zeros(len(df) - len(macd_hist)), macd_hist])
        return indicator_df

class marketfi(Indicator):
    '''
    Market Facilitation Index
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = ti.marketfi(df['high'].values, df['low'].values, df['volume'].values.astype(float))
        return indicator_df

class mass(Indicator):
    '''
    Mass Index
    '''
    def __init__(self, period = 9):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.mass(df['high'].values, df['low'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df

class mfi(Indicator):
    '''
    Money Flow Index
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.mfi(df['high'].values, df['low'].values, df['close'].values, df['volume'].values.astype(float), self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df


class mom(Indicator):
    '''
    Momentum
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.mom(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class msw_sine(Indicator):
    '''
    Mesa Sine Wave
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        msw, msw_signal = ti.msw(df[self.key].values, self.period)
        indicator_df[self.get_default_label() + "_sine"] = np.concatenate([np.zeros(len(df) - len(msw)), msw])
        return indicator_df
    
class msw_lead(Indicator):
    '''
    Mesa Sine Wave
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        msw, msw_signal = ti.msw(df[self.key].values, self.period)
        indicator_df[self.get_default_label() + "_lead"] = np.concatenate([np.zeros(len(df) - len(msw_signal)), msw_signal])
        return indicator_df
    
class natr(Indicator):
    '''
    Normalized Average True Range
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.natr(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class nvi(Indicator):
    '''
    Negative Volume Index
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = ti.nvi(df['close'].values, df['volume'].values.astype(float))
        return indicator_df
    
class obv(Indicator):
    '''
    On-Balance Volume
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = ti.obv(df['close'].values, df['volume'].values.astype(float))
        return indicator_df
    
class ppo(Indicator):
    '''
    Percentage Price Oscillator
    '''
    def __init__(self, key='close', short = 12, long = 26):
        self.key = key
        self.short = short
        self.long = long
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.ppo(df[self.key].values, self.short, self.long)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class pvi(Indicator):
    '''
    Positive Volume Index
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = ti.pvi(df['close'].values, df['volume'].values.astype(float))
        return indicator_df

class qstick(Indicator):
    '''
    Qstick
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.qstick(df['open'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class roc(Indicator):
    '''
    Rate of Change
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.roc(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class rocr(Indicator):
    '''
    Rate of Change Ratio
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.rocr(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class rsi(Indicator):
    '''
    Relative Strength Index
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.rsi(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class stoch_k(Indicator):
    '''
    Stochastic Oscillator
    '''
    def __init__(self, period = 14, smoothK = 3, smoothD = 3):
        self.period = period
        self.smoothK = smoothK
        self.smoothD = smoothD
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        stoch_k, stoch_d = ti.stoch(df['high'].values, df['low'].values, df['close'].values, self.period, self.smoothK, self.smoothD)
        indicator_df[self.get_default_label() + "_k"] = np.concatenate([np.zeros(len(df) - len(stoch_k)), stoch_k])
        return indicator_df
    
class stoch_d(Indicator):
    '''
    Stochastic Oscillator
    '''
    def __init__(self, period = 14, smoothK = 3, smoothD = 3):
        self.period = period
        self.smoothK = smoothK
        self.smoothD = smoothD
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        stoch_k, stoch_d = ti.stoch(df['high'].values, df['low'].values, df['close'].values, self.period, self.smoothK, self.smoothD)
        indicator_df[self.get_default_label() + "_d"] = np.concatenate([np.zeros(len(df) - len(stoch_d)), stoch_d])
        return indicator_df
    
class stochrsi(Indicator):
    '''
    Stochastic RSI
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.stochrsi(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class tr(Indicator):
    '''
    True Range
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        indicator_df[self.get_default_label()] = ti.tr(df['high'].values, df['low'].values, df['close'].values)
        return indicator_df
    
class trix(Indicator):
    '''
    Technical Analysis Indicator
    '''
    def __init__(self, key='close', period = 14):
        self.key = key
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.trix(df[self.key].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class ultosc(Indicator):
    '''
    Ultimate Oscillator
    '''
    def __init__(self, short = 7, medium = 14, long = 28):
        self.short = short
        self.medium = medium
        self.long = long
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.ultosc(df['high'].values, df['low'].values, df['close'].values, self.short, self.medium, self.long)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class vhf(Indicator):
    '''
    Vertical Horizontal Filter
    '''
    def __init__(self, period = 28):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.vhf(df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class volatility(Indicator):
    '''
    Volatility
    '''
    def __init__(self, period = 14):
        self.period = period
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.volatility(df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class vosc(Indicator):
    '''
    Volume Oscillator
    '''
    def __init__(self, short = 12, long = 26):
        self.short = short
        self.long = long
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.vosc(df['volume'].values.astype(float), self.short, self.long)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class wad(Indicator):
    '''
    Williams Accumulation/Distribution
    '''
    def __init__(self):
        pass
        
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.wad(df['high'].values, df['low'].values, df['close'].values)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df
    
class willr(Indicator):
    '''
    Williams %R
    '''
    def __init__(self, period = 14):
        self.period = period
    
    def calculate(self, df):
        indicator_df = pd.DataFrame(index=df.index)
        result = ti.willr(df['high'].values, df['low'].values, df['close'].values, self.period)
        indicator_df[self.get_default_label()] = np.concatenate([np.zeros(len(df) - len(result)), result])
        return indicator_df