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
import pickle



class nn():
    
    def __init__(self, samples, test_samples, sample_size, method, \
    num_layer, num_hidden, timesteps, future_time, batch_size, \
    learning_rate, training_steps, display_step):
        
        self.samples = samples
        self.test_samples = test_samples
        self.sample_size = sample_size
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

            self.tf_train_samples = tf.placeholder("float", [None, self.timesteps],name='x')
            self.tf_train_future_vol = tf.placeholder("float", [None],name='y')
        

            def weight_variable(shape):
                initial = tf.truncated_normal(shape,stddev=0.1)
                return tf.Variable(initial)

            def bias_variable(shape):
                initial = tf.constant(1.0,shape=shape)
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

                w_end = weight_variable([1024,1])
                b_end = bias_variable([1])
                return tf.matmul(h_fc,w_end)+b_end

            def model_lstm(x):
                x = tf.expand_dims(x,2)
                #x = tf.unstack(x,self.timesteps,1)
                w_end = weight_variable([self.num_hidden,1])
                b_end = bias_variable([self.batch_size,1])
                def one_layer():
                    cell = rnn.BasicLSTMCell(num_units=self.num_hidden)
                    cell = rnn.DropoutWrapper(cell,output_keep_prob=0.9)
                    return cell
                cells = [one_layer() for _ in range(self.num_layer)]
                mlstm_cell = rnn.MultiRNNCell(cells,state_is_tuple=True)
                init_state = mlstm_cell.zero_state(self.batch_size,dtype=tf.float32)

                outputs, _ = tf.nn.dynamic_rnn(mlstm_cell,x,initial_state=init_state)
                return tf.matmul(outputs[:,-1,:],w_end)+b_end
            
            def model_basic_nn(x):
                dense = x
                for layer in range(self.num_layer):
                    num_hidden = self.num_hidden/2
                    dense = tf.layers.dense(dense,num_hidden,activation=tf.nn.relu)
                dense = tf.layers.dense(dense,1)
                return dense
            
            if self.method == 'cnn':
                self.output = model_cnn(self.tf_train_samples)
            elif self.method == 'lstm':
                self.output = model_lstm(self.tf_train_samples)
            elif self.method == 'bnn':
                self.output = tf.identity(model_basic_nn(self.tf_train_samples),name='out')
            
            #y = tf.one_hot(self.tf_train_future_vol,5)
            
            self.prediction = tf.less(self.output,self.tf_train_samples[:,-1])
            self.true_label = tf.less(self.tf_train_future_vol,self.tf_train_samples[:,-1])
            correct_prediction = tf.equal(self.prediction, self.true_label)
            #self.rough_prediction = tf.maximum(tf.minimum(self.prediction,3),1)
            #self.rough_label = tf.maximum(tf.minimum(self.true_label,3),1)
            #rough_correct_prediction = tf.equal(self.rough_prediction,self.rough_label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            #self.rough_accuracy = tf.reduce_mean(tf.cast(rough_correct_prediction,"float"))
            #self.loss = -tf.reduce_mean(y * tf.log(output))
            #self.loss = -tf.reduce_mean(tf.one_hot(self.rough_label,3)*tf.log(output))
            self.mean_loss = tf.reduce_mean(tf.square(tf.subtract(tf.reduce_mean(self.tf_train_samples,1),self.tf_train_future_vol)))
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.output,self.tf_train_future_vol)))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.saver = tf.train.Saver()

    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:

            tf.global_variables_initializer().run()
            
            print("Start:")

            l = np.zeros(self.training_steps)
            l_test = np.zeros(self.training_steps)
            l_mean = np.zeros(self.training_steps)
            accuracy = np.zeros(self.training_steps)
            #accuracy = np.zeros(self.training_steps)
            #rough_accuracy = np.zeros(self.training_steps)
            predict_value = []
            test_x = self.test_samples[0]
            test_y = self.test_samples[1]

            for step in range(0, self.training_steps):
                
                training_x, training_y = get_chunk(step,self.batch_size,self.samples,self.sample_size,self.timesteps)
                # Run optimization op (backprop)
                _, l[step], predict_value = sess.run([self.optimizer,self.loss,self.output], \
                feed_dict={self.tf_train_samples: training_x, self.tf_train_future_vol: training_y}) 
                l_test[step], accuracy[step], l_mean[step] = sess.run([self.loss,self.accuracy,self.mean_loss],feed_dict={self.tf_train_samples: test_x, \
                                        self.tf_train_future_vol: test_y})
                if step % self.display_step == 0:
                    
                    print("Step " + str(step) + ", Loss= " + format(l[step]))
                    #print(training_y)
                    #print(np.mean(training_x,1))
                    #print(predict_value)
                    print(accuracy[step])

            self.saver.save(sess, './checkpoint_dir/MyModel')

            plt.figure()
            f_name = 'batch_size_'+str(self.batch_size)+'num_layer_'+str(self.num_layer)+'.png'
            plt.plot(l)
            plt.plot(l_test)
            plt.plot(l_mean)
            plt.savefig(f_name)
            print("Optimization Finished!")
    
    def test(self,data_file):

        test_data = pd.read_pickle(data_file)
        length = len(test_data[0])
        prediction = np.zeros(length)
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        output = graph.get_tensor_by_name('out:0')
        for i in range(10):
            a = np.array(test_data[1][i])
            a = a.reshape([1,])
            prediction[i] = sess.run(output,feed_dict={x: test_data[0][i][np.newaxis,:], y: a})
            print(i/length)
        
        
        out = open('2009_prediction.pkl','wb')
        pickle.dump(prediction,out,-1)
        out.close()
        


