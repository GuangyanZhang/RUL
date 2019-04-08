# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:08:48 2019

@author: ZGY
"""

from pandas import DataFrame
from pandas import concat
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
data1 = pd.read_csv('F:/alibaba/Learning_set/Bearing1_1/bear1.csv')
data7 = pd.read_csv('F:/alibaba/Learning_set/Bearing1_7/bear7.csv')
data6 = pd.read_csv('F:/alibaba/Learning_set/Bearing1_6/bear6.csv')
data2 = pd.read_csv('F:/alibaba/Learning_set/Bearing1_2/bear2.csv')
train_data = data1.append([data2,data6,data7])
time2sup_data = series_to_supervised(data=list(train_data['fft_mean'].values),n_in=300,dropnan=True) 

data3 = pd.read_csv('F:/alibaba/Test_set/Bearing1_3/bear3.csv')
time2sup3 = series_to_supervised(data=list(data3['fft_mean'].values),n_in=300,dropnan=True) 




gbm0 = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=64,
    learning_rate=0.1,
    n_estimators=10000)

print(time2sup_data.shape)
x_train = time2sup_data
x_test = time2sup3

print(x_train.shape)
print(x_test.shape)

y_label = x_test['var1(t)']
y_train = x_train.pop('var1(t)')
y_test = x_test.pop('var1(t)')


# 损失函数mse
gbm0.fit(x_train.values,y_train,eval_set=[(x_test.values,y_test)],eval_metric='mse',early_stopping_rounds=15)
print(gbm0.predict(x_test.values))
print('mse',sum(y_label-gbm0.predict(x_test.values))/len(y_label))
y_veiw = gbm0.predict(x_test.values)
y_predict = []

for i in range(1200):
    if i==0:
        x_last = np.array(x_test[-1:])
        x_last = x_last.tolist()
        y_pre = y_test[-1:].tolist()
        x_last[0] = x_last[0] + y_pre
        x_last[0].pop(0)
    else:
        y_pre = y_pre.tolist()
        x_last[0] = x_last[0] + y_pre
        x_last[0].pop(0)
    y_pre = gbm0.predict(x_last)
    y_predict.append(y_pre)
    

line1 = plt.plot(range(len(x_test)),gbm0.predict(x_test.values),label=u'predict')
line2 = plt.plot(range(len(y_test)),y_test.values,label=u'true')
plt.plot(range(len(y_test),len(y_test)+len(y_predict)),y_predict,label=u'predict_rul')
Threshold = []
for i in range(len(y_test)+len(y_predict)):
    Threshold.append(40)

plt.plot(range(len(y_test)+len(y_predict)),Threshold,label=u'Threshold')
plt.legend()
plt.show()