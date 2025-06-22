import indicators as inds
from indicators import pandas_ta_indicators as pti
from indicators import tulipy_indicators as ti
from indicators import custom_indicators as ci
from indicators import metaindicators as mi

#### FEATURE SETS ####
all_tulipy_defaults = [
    ti.adosc(), ti.adx(), ti.adxr(),
    ti.ao(), ti.apo(), ti.aroon_up(), ti.aroon_down(), ti.aroonosc(),
      ti.bop(),
    ti.cci(), ti.cmo(), ti.cvi(),
    ti.di_plus(), ti.di_minus(),  ti.dpo(), 
    ti.dx(), ti.emv(), ti.fisher(), ti.fosc(), ti.kvo(), ti.linregslope(), 
    ti.macd(), ti.macd_signal(), ti.macd_hist(),
    ti.marketfi(), ti.mass(), ti.mfi(),
    ti.mom(), ti.msw_sine(), ti.msw_lead(), ti.natr(), 
    ti.ppo(),  ti.qstick(), ti.roc(),
    ti.rocr(), ti.rsi(), ti.stoch_k(), ti.stoch_d(),
    ti.trix(), ti.ultosc(), ti.vhf(), 
    ti.volatility(), ti.vosc(),  ti.willr(),
    ti.obv()
] + [
    #the following cols are not stationary by default, so need to be modified
    mi.divide(ind, mi.EMA(ind, period=14)) for ind in [ti.atr(), ti.dm_plus(), ti.dm_minus(), ti.tr(),ti.bbands_upper(),
                                            ti.bbands_middle(), ti.bbands_lower(), ti.ad(), ti.dema(), 
                                            ti.linregintercept(), ti.nvi(), ti.pvi(), ti.wad()]
]

all_candlestick_patterns = [
    pti.all_candlestick_patterns()
]

all_custom_indicators = [
    ci.SMA(), ci.EMA(), ci.tears_bottom(), ci.normalized_relative_price(), ci.percent_from_local_high(), ci.percent_from_local_low()
]

periods = [1,2,3,5,10,14,21,30,50,100,200,330]
sma_ratios_close = [
    mi.divide(inds.SMA(val1), inds.SMA(val2)) for val1 in periods for val2 in periods if val1 < val2
]
ema_ratios_close = [
    mi.divide(inds.EMA(val1), inds.EMA(val2)) for val1 in periods for val2 in periods if val1 < val2
]
sma_ratios_vwap = [
    mi.divide(inds.SMA(val1, key='vwap'), inds.SMA(val2, key='vwap')) for val1 in periods for val2 in periods if val1 < val2
]
ema_ratios_vwap = [
    mi.divide(inds.EMA(val1, key='vwap'), inds.EMA(val2, key='vwap')) for val1 in periods for val2 in periods if val1 < val2
]
perc_from_extrema = [
    ci.percent_from_local_high(period=p) for p in periods]+[
    ci.percent_from_local_low(period=p) for p in periods
]


z_score_close_volume = [
    mi.rolling_znorm(inds.ticker(key='close'), period) for period in periods]+[
    mi.rolling_znorm(inds.ticker(key='volume'), period) for period in periods
]

test_set = [ci.normalized_relative_price()]


F001 = all_tulipy_defaults + ema_ratios_close + [ci.percent_from_local_high(), ci.percent_from_local_low()]
F002 = [ti.kvo(), ti.volatility(), mi.divide(inds.EMA(3), inds.EMA(10)), ti.di_minus(), mi.divide(inds.EMA(2), inds.EMA(3)),
        ti.rocr(), ti.apo(), ti.natr(), mi.divide(ci.ticker(), ti.dema()), ti.macd_signal()] #top 10 from F001 based on random forest importance



#### TICKER SETS ####
T001 = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK-B', 'JPM', 'LLY', 'COP', 'COST', 'XOM', 'VZ', 'IAU', 'GE']