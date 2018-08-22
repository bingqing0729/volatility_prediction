from class_nn import nn
import sys
import os
from train_test_set import train_test_set

if __name__ == '__main__':

    year = 'samples_2009.pkl'
    train_samples, test_samples, m = train_test_set(year)

    net = nn(samples = train_samples, test_samples = test_samples, sample_size = m,  \
    method = 'bnn', num_layer = int(5), \
    num_hidden = int(512), timesteps = 20, future_time = 10, batch_size = int(sys.argv[1]), \
    learning_rate = float(0.0001), training_steps = 5000, display_step = 100)
    net.define_graph()
    net.run()
    net.test(year)