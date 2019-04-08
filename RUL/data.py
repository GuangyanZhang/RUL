# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:06:18 2019

@author: ZGY
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.fftpack import fft,ifft
df_time = []
df_fft  = []
bear1 = pd.DataFrame(columns = ['time_mean','fft_mean']) 
file_name = os.listdir('F:/alibaba/Learning_set/Bearing1_7')

for name in file_name:
    df = pd.read_csv('F:/alibaba/Learning_set/Bearing1_7/' + str(name), header=None)
    mean_fft = abs(fft(df[4])).mean()
    mean = abs(df[4]).mean()
    df_time.append(mean)
    df_fft.append(mean_fft)

bear1['time_mean'] = df_time
bear1['fft_mean'] = df_fft
bear1.to_csv('F:/alibaba/Learning_set/Bearing1_7/bear7.csv')
plt.subplot(211) 
plt.plot(bear1['time_mean'],label=u'time')
plt.legend()
plt.show()
plt.subplot(212) 
Threshold = []
yf=abs(bear1['fft_mean']) 
for i in range(len(yf)):
    Threshold.append(40)
plt.plot(range(len(yf)),Threshold,label=u'Threshold')
plt.plot(yf,label=u'fft')
plt.legend()
plt.show()