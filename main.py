from class_lstm import cnn

if __name__ == '__main__':
    
    net = cnn(num_hidden = 5, timesteps = 50, future_time = 10, batch_size = 10, \
    learning_rate = 0.00001, training_steps = 1000, display_step = 100)
    net.define_graph()
    net.run()
