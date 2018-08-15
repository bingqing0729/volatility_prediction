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
n_day = 250
n_stock = amt_return.shape[1]

x = np.zeros([n_day*n_stock,200,6])
y = np.zeros(n_day*n_stock)

i = 0
start = sum(amt_return.index<20060101)

for date in range(0,n_day):
    for stock in range(0,n_stock):
        print(i)
        x[i,:,0] = list(amt_return.iloc[start+date:start+date+200,stock]/10)
        x[i,:,1] = list(open_return.iloc[start+date:start+date+200,stock])
        x[i,:,2] = list(high_return.iloc[start+date:start+date+200,stock])
        x[i,:,3] = list(low_return.iloc[start+date:start+date+200,stock])
        x[i,:,4] = list(close_return.iloc[start+date:start+date+200,stock])
        temp = np.std(close_return.iloc[start+date+200-10:start+date+200,stock])
        if temp == 0:
            continue
        y[i] = (np.std(close_return.iloc[start+date+200:start+date+210,stock]) -temp)/temp
        x[i,:,5] = y[i]
        i = i+1

t = [t for t in x if not np.any(np.isnan(t))]
t = [j for j in t if not np.all(j[:,0:5]==0)]
x = [a[:,0:5] for a in t]
y = [a[0,5] for a in t]
y = np.array(y)
#y[y>0.08] = 4
#y[(y>0.02)*(y<=0.08)] = 3
#y[(y>-0.02)*(y<=0.02)] = 2
#y[(y>-0.08)*(y<=-0.02)] = 1
#y[y<=-0.08] = 0

r = [x,y]

filename = 'samples/samples_2006.pkl'
output = open(filename,'wb')
pickle.dump(r,output,-1)
output.close()