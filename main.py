import pandas as pd
from class_nn import nn
import sys
import os
import random

if __name__ == '__main__':
    year = 's_samples_2009.pkl'
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
    net = nn(samples = train_samples, test_samples = test_samples, sample_size = m,  \
    method = 'bnn', num_layer = int(5), \
    num_hidden = int(512), timesteps = 20, future_time = 10, batch_size = int(sys.argv[1]), \
    learning_rate = float(0.0001), training_steps = 5000, display_step = 100)
    net.define_graph()
    net.run()
