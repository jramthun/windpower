# %%
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

df = pd.read_csv('./Turbine_Data.csv')
# df = pd.read_csv('./WindPower/Turbine_Data.csv')
# print(df.head())
df['Datetime'] = pd.to_datetime(df['Datetime'])
# print(df.head())
df = df.resample('D', on='Datetime', origin='start').mean()

# print(df.head())
# df = df.loc['2019-10-30':]
df.index = pd.DatetimeIndex(df.index).to_period('D')
# print(df.head())
df = df[['ActivePower']]
# print(df.head())

# SARIMAX: Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model
train_split = 0.9
dfl = len(df)
# print(dfl)
tra = df[0:int(dfl*train_split)].dropna()
tes = df[int(dfl*train_split):].dropna()
# print(tra.index[-1])
# print(tes.index[0], tes.index[-1])
# print(tra.tail())
# print(tre.head())
tr_end = tra.index[-1]
te_srt, te_end = tes.index[0], tes.index[-1]
# print(tes.index)

# %%
df.index = df.index.to_timestamp()
import matplotlib.pyplot as plt
import statsmodels.api as sm
fig,ax = plt.subplots(2,1,figsize=(20,10))
fig = sm.graphics.tsa.plot_acf(tra.diff().dropna(), lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(tra.diff().dropna(), lags=50, ax=ax[1], method='ywm')
plt.show()

res = sm.tsa.adfuller(df['ActivePower'].dropna(), regression='ct')
print('p-value:{}'.format(res[1]))

res = sm.tsa.adfuller(df['ActivePower'].diff().dropna(), regression='c')
print('p-value:{}'.format(res[1]))


res = sm.tsa.seasonal_decompose(df['ActivePower'].dropna(), period=28)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()

# %%
resDiff = sm.tsa.arma_order_select_ic(tra, max_ar=30, max_ma=30, ic='aic', trend='c')
print('ARMA(p,q) =',resDiff['aic_min_order'],'is the best.')
# Saved Result: ARMA(p,q) = (4,3) is the best.
# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
p, q, d = 4, 1, 3
P, Q, D, s = p, q, d, 26
sarimax = SARIMAX(endog=tra, order=(p, q, d), seasonal_order=(P, Q, D, s), freq='D').fit()
sarimax.summary()
sarimax.plot_diagnostics(figsize=(15, 12))

res = sarimax.resid
fig, ax = plt.subplots(2, 1, figsize=(15, 8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1], method='ywm')
plt.show()

pred = sarimax.predict(te_srt, te_end+1)[1:]
print(len(tes), len(pred))
print('SARIMAX model MSE:{}'.format(mean_squared_error(tes, pred)))

tes.plot()
pred.plot()
plt.show()


# %%
from statsmodels.tsa.arima.model import ARIMA
arima = ARIMA(endog=tra, order=(4,1,3), freq='D').fit()
arima.summary()
arima.plot_diagnostics(figsize=(15, 12))

res = arima.resid
fig, ax = plt.subplots(2, 1, figsize=(15, 8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1], method='ywm')
plt.show()

apred = arima.predict(te_srt, te_end+1)[1:]
# print(len(tes), len(apred))
print('ARIMAX model MSE:{}'.format(mean_squared_error(tes, apred)))
tes.plot()
apred.plot()
plt.show()

# %%
