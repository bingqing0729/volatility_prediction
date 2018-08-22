import numpy as np
import pandas as pd
import pickle

close_return = pd.read_pickle('close_return.pkl')
n_day = 250
n_stock = close_return.shape[1]
x = []
y = []
d = []
s = []
start = sum(close_return.index<20090101)
i = 0
for date in range(n_day):
    for stock in range(n_stock):
        print(i)
        
        a = [np.std(close_return.iloc[start+date-10-t:start+date-t,stock]) for t in range(20)]
        a.reverse()
        a = np.array(a)
        if not np.any(np.isnan(a)):
            x.append(a)
            b = np.std(close_return.iloc[start+date:start+date+10,stock])
            y.append(b)
            d.append(close_return.index[start+date])
            s.append(close_return.columns[stock])
            i = i + 1

x = np.array(x)
y = np.array(y)
r = [[],[],[],[]]
r[0] = x[y<=1]
r[1] = y[y<=1]
r[2] = d[y<=1]
r[3] = s[y<=1]

filename = 'samples_2009.pkl'
output = open(filename,'wb')
pickle.dump(r,output,-1)
output.close()
