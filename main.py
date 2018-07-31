from class_nn import nn
import sys

if __name__ == '__main__':
    
    net = nn(method = 'cnn', num_layer = 3, \
    num_hidden = 5, timesteps = 200, future_time = 10, batch_size = 100, \
    learning_rate = 0.0001, training_steps = 1000, display_step = 100)
    net.define_graph()
    net.run()
