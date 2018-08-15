import numpy as np
import pandas as pd
import pickle

close_return = pd.read_pickle('close_return.pkl')
n_day = 10
n_stock = close_return.shape[1]
x = []
y = []
start = sum(close_return.index<20090101)
for date in range(n_day):
    for stock in range(n_stock):
        a = [np.std(close_return.iloc[start+date-10-i:start+date-i,stock]) for i in range(20)]
        a.reverse()
        a = np.array(a)
        if not np.any(np.isnan(a)):
            x.append(a)
            b = np.std(close_return.iloc[start+date:start+date+10,stock])
            y.append(b)
r = [x,y]

filename = 's_samples_2009.pkl'
output = open(filename,'wb')
pickle.dump(r,output,-1)
output.close()