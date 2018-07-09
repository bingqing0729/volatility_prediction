from __future__ import print_function
from get_chunk import get_chunk
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import cudnn_rnn
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



class cnn():
    
    def __init__(self, timesteps, future_time, batch_size, num_hidden, learning_rate, training_steps, display_step):
        
        self.timesteps = timesteps
        self.future_time = future_time
        self.batch_size = batch_size
        self.display_step = display_step
        
        # Hyper parameters
        self.training_steps = training_steps
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        
        # Graph related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_future_vol = None



    def define_graph(self):
        
        with self.graph.as_default():
            self.tf_train_samples = tf.placeholder("float", [self.batch_size, self.timesteps, 5])
            self.tf_train_future_vol = tf.placeholder("float", [self.batch_size])
            
            def weight_variable(shape):
                initial = tf.truncated_normal(shape,stddev=0.1)
                return tf.Variable(initial)

            def bias_variable(shape):
                initial = tf.constant(0.1,shape=shape)
                return tf.Variable(initial)

            def conv2d(x,W):
                return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
            
            def max_pool_2x2(x):
                return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            
            def model(x):

                W_conv = weight_variable([5,5,1,32])
                b_conv = bias_variable([32])

                x = tf.reshape(x,[self.batch_size,self.timesteps,5,1])
                h_conv = tf.nn.relu(conv2d(x,W_conv)+b_conv)
                h_pool = max_pool_2x2(h_conv)

                w_fc = weight_variable([25*3*32,64])
                b_fc = bias_variable([64])

                h_pool_flat = tf.reshape(h_pool,[self.batch_size,25*3*32])
                h_fc = tf.nn.relu(tf.matmul(h_pool_flat,w_fc)+b_fc)

                w_end = weight_variable([64,1])
                b_end = bias_variable([1])
                return tf.matmul(h_fc,w_end)+b_end

            output = model(self.tf_train_samples)
            output = tf.abs(output)
            self.loss = tf.reduce_sum(tf.square(output-self.tf_train_future_vol))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            self.vol = tf.reduce_mean(output)
            self.vol_true = tf.reduce_mean(self.tf_train_future_vol)

    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:

            tf.global_variables_initializer().run()
            
            print("Start:")

            l = np.zeros(self.training_steps)
            vol = np.zeros(self.training_steps)
            vol_true = np.zeros(self.training_steps)

            for step in range(0, self.training_steps):
                
                training_x, training_y = get_chunk(self.batch_size,self.timesteps,self.future_time)
                # Run optimization op (backprop)
                _, l[step], vol[step], vol_true[step] = sess.run([self.optimizer,self.loss,self.vol,self.vol_true], \
                feed_dict={self.tf_train_samples: training_x, self.tf_train_future_vol: training_y})       
                if step % self.display_step == 0:
                    print("Step " + str(step) + ", Loss= " + format(l[step]) \
                     + ", vol_mean= " + format(vol[step]) + ", vol_true_mean= " + format(vol_true[step]))

            print("Optimization Finished!")



