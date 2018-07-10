import numpy as np
import pandas as pd
import pickle

# import ohlc and amount data
amt_return = pd.read_pickle('amt_return.pkl')
high_return = pd.read_pickle('high_return.pkl')
low_return = pd.read_pickle('low_return.pkl')
open_return = pd.read_pickle('open_return.pkl')
close_return = pd.read_pickle('close_return.pkl')

#n_day = amt_return.shape[0]-200-10
n_day = 100
n_stock = amt_return.shape[1]

x = np.zeros([n_day*n_stock,200,6])
y = np.zeros(n_day*n_stock)

i = 0

for date in range(0,n_day):
    for stock in range(0,n_stock):
        x[i,:,0] = list(amt_return.iloc[date:date+200,stock]/10)
        x[i,:,1] = list(open_return.iloc[date:date+200,stock])
        x[i,:,2] = list(high_return.iloc[date:date+200,stock])
        x[i,:,3] = list(low_return.iloc[date:date+200,stock])
        x[i,:,4] = list(close_return.iloc[date:date+200,stock])
        y[i] = (np.std(close_return.iloc[date+200:date+210,stock]) \
        -np.std(close_return.iloc[date+200-10:date+200,stock]))*np.sqrt(250/10)
        x[i,:,5] = y[i]
        i = i+1

t = [t for t in x if not np.any(np.isnan(t))]
x = [a[:,0:5] for a in t]
y = [a[0,5] for a in t]
y = np.array(y)
y[y>0.08] = 4
y[(y>0.02)*(y<=0.08)] = 3
y[(y>-0.02)*(y<=0.02)] = 2
y[(y>-0.08)*(y<=-0.02)] = 1
y[y<=-0.08] = 0

r = [x,y]

filename = 'samples_timestep_200.pkl'
output = open(filename,'wb')
pickle.dump(r,output,-1)
output.close()