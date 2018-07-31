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



class nn():
    
    def __init__(self, method, num_layer, num_hidden, timesteps, future_time, batch_size, learning_rate, training_steps, display_step):
        
        self.method = method
        self.timesteps = timesteps
        self.future_time = future_time
        self.batch_size = batch_size
        self.display_step = display_step
        
        # Hyper parameters
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        
        # Graph related
        self.graph = tf.Graph()
        self.num_layer = num_layer
        self.num_hidden = num_hidden
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

            
            def model_cnn(x):

                x = tf.reshape(x,[self.batch_size,self.timesteps,5,1])
                for j in range(0,self.num_layer):
                    num_core = self.num_hidden*(j+1)
                    W_conv = weight_variable([2,2,1,num_core])
                    b_conv = bias_variable([num_core])
                    h_conv = tf.nn.relu(conv2d(x,W_conv)+b_conv)


                w_fc = weight_variable([self.timesteps*5*num_core,1024])
                b_fc = bias_variable([1024])

                h_pool_flat = tf.reshape(h_conv,[self.batch_size,self.timesteps*5*num_core])
                h_fc = tf.nn.relu(tf.matmul(h_pool_flat,w_fc)+b_fc)

                w_end = weight_variable([1024,5])
                b_end = bias_variable([5])
                return tf.nn.softmax(tf.matmul(h_fc,w_end)+b_end)

            def model_lstm(x):
                
                #x = tf.unstack(x,self.timesteps,1)
                w_end = weight_variable([self.num_hidden,5])
                b_end = bias_variable([5])
                def one_layer():
                    cell = rnn.BasicLSTMCell(num_units=self.num_hidden)
                    cell = rnn.DropoutWrapper(cell,output_keep_prob=0.9)
                    return cell
                cells = [one_layer() for _ in range(self.num_layer)]
                mlstm_cell = rnn.MultiRNNCell(cells,state_is_tuple=True)
                init_state = mlstm_cell.zero_state(self.batch_size,dtype=tf.float32)

                outputs, _ = tf.nn.dynamic_rnn(mlstm_cell,x,initial_state=init_state)
                return tf.nn.softmax(tf.nn.relu(tf.matmul(outputs[:,-1,:],w_end)+b_end))


            if self.method == 'cnn':
                output = model_cnn(self.tf_train_samples)
            else:
                output = model_lstm(self.tf_train_samples)
            
            y = tf.one_hot(self.tf_train_future_vol,5)
            prediction = tf.argmax(output,1)
            true_label = tf.argmax(y,1)
            correct_prediction = tf.equal(prediction, true_label)
            rough_prediction = tf.maximum(tf.minimum(prediction,1),3)
            rough_label = tf.maximum(tf.minimum(true_label,1),3)
            rough_correct_prediction = tf.equal(rough_prediction,rough_label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.rough_accuracy = tf.reduce_mean(tf.cast(rough_correct_prediction),"float")
            self.loss = -tf.reduce_mean(y * tf.log(output))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:

            tf.global_variables_initializer().run()
            
            print("Start:")

            l = np.zeros(self.training_steps)
            accuracy = np.zeros(self.training_steps)
            rough_accuracy = np.zeros(self.training_steps)

            for step in range(0, self.training_steps):
                
                training_x, training_y = get_chunk(self.batch_size)
                # Run optimization op (backprop)
                _, l[step], accuracy[step], rough_accuracy[step] = sess.run([self.optimizer,self.loss,self.accuracy,self.rough_accuracy], \
                feed_dict={self.tf_train_samples: training_x, self.tf_train_future_vol: training_y})       
                if step % self.display_step == 0:
                    print("Step " + str(step) + ", Loss= " + format(l[step]) \
                     + ", accuracy= " + format(accuracy[step]) + ", rough prediction= " + format(rough_accuracy[step]))
            
            plt.figure()
            f_name = 'batch_size_'+str(self.batch_size)+'num_layer_'+str(self.num_layer)+'.png'
            plt.plot(accuracy)
            plt.savefig(f_name)
            print("Optimization Finished!")



