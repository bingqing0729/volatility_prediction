import pandas as pd
from class_nn import nn
import sys
import os
import random

if __name__ == '__main__':
    year = 's_samples_2009.pkl'
    samples = pd.read_pickle(year)
    m = len(samples[1])
    s = random.sample(range(0,m),10000)
    samples[0]=[samples[0][k] for k in s]
    samples[1]=[samples[1][k] for k in s]
    net = nn(samples = samples, sample_size = m,  method = 'lstm', num_layer = int(1), \
    num_hidden = int(10), timesteps = 20, future_time = 10, batch_size = int(10), \
    learning_rate = float(0.1), training_steps = 2000, display_step = 100)
    net.define_graph()
    net.run()
