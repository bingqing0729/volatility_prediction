from class_cnn import cnn

if __name__ == '__main__':
    
    net = cnn(num_layer = 3, timesteps = 200, future_time = 10, batch_size = 100, \
    learning_rate = 0.0001, training_steps = 5000, display_step = 100)
    net.define_graph()
    net.run()
