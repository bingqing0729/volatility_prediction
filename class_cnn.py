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
            self.tf_train_future_vol = tf.placeholder("int32", [self.batch_size])
            
            def weight_variable(shape):
                initial = tf.truncated_normal(shape,stddev=0.1)
                return tf.Variable(initial)

            def bias_variable(shape):
                initial = tf.constant(0.1,shape=shape)
                return tf.Variable(initial)

            def conv2d(x,W):
                return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

            
            def model(x):

                W_conv = weight_variable([5,5,1,32])
                b_conv = bias_variable([32])

                x = tf.reshape(x,[self.batch_size,self.timesteps,5,1])
                h_conv = tf.nn.relu(conv2d(x,W_conv)+b_conv)

                W_conv2 = weight_variable([5, 5, 32, 64])
                b_conv2 = bias_variable([64])

                h_conv2 = tf.nn.relu(conv2d(h_conv, W_conv2) + b_conv2) 

                w_fc = weight_variable([50*5*64,64])
                b_fc = bias_variable([64])

                h_pool_flat = tf.reshape(h_conv2,[self.batch_size,50*5*64])
                h_fc = tf.nn.relu(tf.matmul(h_pool_flat,w_fc)+b_fc)

                w_end = weight_variable([64,6])
                b_end = bias_variable([6])
                return tf.nn.softmax(tf.matmul(h_fc,w_end)+b_end)

            output = model(self.tf_train_samples)
            y = tf.one_hot(self.tf_train_future_vol,6)
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.loss = -tf.reduce_sum(y * tf.log(output+0.000001))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:

            tf.global_variables_initializer().run()
            
            print("Start:")

            l = np.zeros(self.training_steps)
            accuracy = np.zeros(self.training_steps)

            for step in range(0, self.training_steps):
                
                training_x, training_y = get_chunk(self.batch_size,self.timesteps,self.future_time)
                # Run optimization op (backprop)
                _, l[step], accuracy[step] = sess.run([self.optimizer,self.loss,self.accuracy], \
                feed_dict={self.tf_train_samples: training_x, self.tf_train_future_vol: training_y})       
                if step % self.display_step == 0:
                    print("Step " + str(step) + ", Loss= " + format(l[step]) \
                     + ", accuracy " + format(accuracy[step]))

            print("Optimization Finished!")



