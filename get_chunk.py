import random
import pandas as pd

samples = pd.read_pickle("samples_timestep_200.pkl")
m = len(samples[1])
def get_chunk(n):
    l = random.sample(range(0,m),n)
    x = [samples[0][t] for t in l]
    y = [samples[1][t] for t in l]
    return(x,y)

