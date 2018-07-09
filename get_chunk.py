
import random
import numpy as np
import pandas as pd

# import ohlc and amount data
amt_return = pd.read_pickle('amt_return.pkl')
high_return = pd.read_pickle('high_return.pkl')
low_return = pd.read_pickle('low_return.pkl')
open_return = pd.read_pickle('open_return.pkl')
close_return = pd.read_pickle('close_return.pkl')


def get_chunk(n,timesteps,future_time):
    
    x = np.zeros([n,timesteps,5])
    y = np.zeros(n)
    
    p = 0
    while p < 1:
        p = p+1
        date = np.random.randint(0,len(list(amt_return.index[0:-timesteps-future_time])),1)[0]
        start_date = amt_return.index[date]
        mid_date = amt_return.index[date+timesteps]
        end_date = amt_return.index[date+timesteps+future_time]
        amt_temp = amt_return.iloc[date:date+future_time+timesteps,:]
        amt_temp = amt_temp.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        num_valid_stocks = amt_temp.shape[1]
        if num_valid_stocks < n:
            p = p-1
    stocks = random.sample(range(0,num_valid_stocks),n)

    i = 0
    
    while i < n:
        stock = amt_temp.columns[stocks[i]]
        x[i,:,0] = list(amt_return.loc[start_date:mid_date-1,stock]/10)
        x[i,:,1] = list(open_return.loc[start_date:mid_date-1,stock])
        x[i,:,2] = list(high_return.loc[start_date:mid_date-1,stock])
        x[i,:,3] = list(low_return.loc[start_date:mid_date-1,stock])
        x[i,:,4] = list(close_return.loc[start_date:mid_date-1,stock])
        y[i] = np.std(close_return.loc[mid_date:end_date-1,stock])*np.sqrt(250/future_time)

        i = i+1

    return(x,y)
