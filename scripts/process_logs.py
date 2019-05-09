import numpy as np
from scipy.stats import t
import glob
import pandas as pd

_filter = range(0, 151, 15)

paths = ['histories.npy', 'histories_0.0.npy', 'me_histories_0.0.npy']
names = ['RE-IRL-ECO', 'RE-IRL-N', 'ME-IRL-N']

df = pd.read_csv('taxi_return_0.0.csv')

for i,p in enumerate(paths):
	a = np.load(p, encoding='latin1')
	ret = a[:, :, 1]
	m = np.mean(ret, axis=0).astype('float')
	s = np.var(ret, axis=0).astype('float') ** 0.5
	inter = t.interval(0.95, 40, loc=m, scale=s)[1] - m
	df[names[i]] = m[_filter]
	df[names[i]+'-error'] = inter[_filter]

df.to_csv('taxi_return_0.0_new.csv', index=None)

paths = ['histories_1.npy', 'histories_0.1.npy', 'me_histories_0.1.npy']
names = ['RE-IRL-ECO', 'RE-IRL-N', 'ME-IRL-N']


df = pd.read_csv('taxi_return_0.1.csv')

for i,p in enumerate(paths):
	a = np.load(p, encoding='latin1')
	ret = a[:, :, 1]
	m = np.mean(ret, axis=0).astype('float')
	s = np.var(ret, axis=0).astype('float') ** 0.5
	inter = t.interval(0.95, 40, loc=m, scale=s)[1] - m
	df[names[i]] = m[_filter]
	df[names[i]+'-error'] = inter[_filter]

df.to_csv('taxi_return_0.1_new.csv', index=None)
