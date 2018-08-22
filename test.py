import tensorflow as tf
import pandas as pd
import numpy as np


test_data = pd.read_pickle(data_file)
length = len(test_data[0])
prediction = np.zeros(length)
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    for i in range(length):
        prediction[i] = sess.run([self.output],feed_dict={self.tf_train_samples: test_data[0][i], \
        self.tf_train_future_vol: test_data[1][i]})
        print(prediction[i])