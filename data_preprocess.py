import pandas as pd
import numpy as np
import pickle

var_list = ['amt','high','low','open','close']

for var in var_list:
    file_name = var + '.pkl'
    output_name = var + '_return.pkl'
    temp = pd.read_pickle(file_name).loc[20050104:20180601]
    daily_return = temp.pct_change(1).iloc[1:-1,:]
    output = open(output_name, 'wb')
    pickle.dump(daily_return, output, -1)
    output.close()