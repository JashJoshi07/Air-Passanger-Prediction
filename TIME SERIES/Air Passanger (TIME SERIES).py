#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("C:\\Users\\jashj\\OneDrive\\Desktop\\Datasets\\AirPassengers.csv")


# In[4]:


data.head()


# In[5]:


data.dtypes


# In[6]:


data.index


# In[7]:


from datetime import datetime
con=data['Month']
data['Month']=pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
#check datatype of index
data.index


# ## CONVERT TO TIME SERIES

# In[8]:


ts = data['#Passengers']
ts.head(10)


# In[9]:


ts['1949-09-01']


# In[10]:


from datetime import datetime
ts[datetime(1949,9,1)]


# In[11]:


ts['1949-01-01' : '1949-05-01']


# In[12]:


ts[:'1949-05-01']


# In[13]:


ts['1949']


# ## Checking the stationary

# In[14]:


plt.plot(ts)


# ## Sationary Testing

# In[15]:


from statsmodels.tsa.stattools import adfuller


# In[16]:


dftest = adfuller(ts, autolag='AIC')
dftest


# In[17]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean=timeseries.rolling(12).mean()
    rolstd=timeseries.rolling(12).std()
    #rolmean = pd.rolling_mean(timeseries, window=12)
    #rolstd = pd.rolling_std(timeseries, window=12)
#Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[18]:


test_stationarity(ts)


# # MAKING THE TIME SERIES STATIONARY

# ### TREND

# In[19]:


ts_log = np.log(ts)
plt.plot(ts_log)


# ## SMOOTHING

# In[20]:


moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color= 'red')


# In[21]:


ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)


# In[22]:


ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_moving_avg_diff.head()


# In[23]:


test_stationarity(ts_log_moving_avg_diff)


# In[24]:


expwighted_avg = ts_log.ewm(span=12).mean()
expwighted_avg_diff = ts_log-expwighted_avg
test_stationarity(expwighted_avg_diff)


# ## SEASONALITY (ALONG WITH TREND) 

# In[25]:


ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)


# In[26]:


ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


# In[27]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual= decomposition.resid

plt.subplot(411)
plt.plot(ts_log , label = 'Originals')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend , label = 'trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal , label = 'seasonal')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual , label = 'residual')
plt.legend(loc='best')

plt.tight_layout()


# In[53]:


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
plt.figure(figsize=(20,10))
test_stationarity(ts_log_decompose)


# ## FORECASTING A TIME SERIES 

# ### ACF & PACF

# In[29]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ts_log_diff, lags =20)
plot_pacf(ts_log_diff, lags =20)
plt.figure(figsize=(20,10))
plt.show()


# In[30]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMA
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf  

lag_acf = acf(ts_log_diff, nlags=5)
lag_pacf = pacf(ts_log_diff, nlags=5, method='ols')

plt.figure(figsize=(20,10))
#Plot ACF:    
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[113]:


result.summary()


# In[32]:


## AR MODEL
import statsmodels.api as sm


# In[39]:


model = sm.tsa.arima.ARIMA(ts_log, order=(1,1,2))
result = model.fit()
plt.figure(figsize=(20,10))
plt.plot(ts_log_diff)
plt.plot(result.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((result.fittedvalues - ts_log_diff)**2))


# In[45]:


#MA model
model = sm.tsa.arima.ARIMA(ts_log, order=(0,1,2))
results_MA = model.fit()
plt.figure(figsize=(20,10))
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues - ts_log_diff)**2))


# In[49]:


#ARIMA model
model = sm.tsa.arima.ARIMA(ts_log, order=(2,1,2))
results_ARIMA = model.fit()
plt.figure(figsize=(20,10))
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - ts_log_diff)**2))


# In[50]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
print(predictions_ARIMA_diff.head())


# In[51]:


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# In[52]:


predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index = ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value = 0)
predictions_ARIMA_log.head()


# In[ ]:





# In[ ]:





# In[54]:


#### Auto arima


# In[55]:


from pmdarima.arima import auto_arima


# In[56]:


modelA = auto_arima(ts, start_p=1, start_q=1,max_p=5,max_d=5,max_q=5)
print (modelA.summary())


# In[57]:


modelA.plot_diagnostics()


# In[58]:


modelA.summary()


# In[ ]:





# In[ ]:




