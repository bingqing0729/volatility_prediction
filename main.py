from class_cnn import cnn

if __name__ == '__main__':
    
    net = cnn(num_hidden = 5, timesteps = 50, future_time = 10, batch_size = 100, \
    learning_rate = 0.000001, training_steps = 500, display_step = 10)
    net.define_graph()
    net.run()
