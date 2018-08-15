import random
import pandas as pd
import numpy as np

def get_chunk(step,n,samples,m,timesteps):
    #l = random.sample(range(0,10000),n)
    if (step+1)%(int(10000/n))==0:
        a = 10000
    else:
        a = 0
    l = range((step*n)%10000,(step+1)*n%10000+a)
    x = np.array([samples[0][t]for t in l])
    y = np.array([samples[1][t] for t in l])
    x[np.isneginf(x)]=-1
    x[np.isposinf(x)]=1
    y[np.isnan(y)] = 0
    return(x,y)
