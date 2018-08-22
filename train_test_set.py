import random
import pandas as pd

def train_test_set(year):    
    samples = pd.read_pickle(year)
    m = len(samples[1])
    s_total = random.sample(range(0,m),20000)
    s = s_total[0:18000]
    s_test = s_total[18000:20000]
    train_samples = [[],[]]
    test_samples = [[],[]]
    train_samples[0]=[samples[0][k] for k in s]
    train_samples[1]=[samples[1][k] for k in s]
    test_samples[0]=[samples[0][k] for k in s_test]
    test_samples[1]=[samples[1][k] for k in s_test]
    return train_samples, test_samples, m