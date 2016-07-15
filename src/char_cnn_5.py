import tensorflow as tf
import numpy as np
from load import load
from collections import defaultdict
import random

epoch_count = 10
batch_size = 128

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
max_text_len = 1014

class_num, train_set, test_set = load("AG", alphabet, max_text_len)

# char cnn
def weight_init(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.02, dtype=tf.float32)
    return tf.Variable(initial)

# batch x max_text_len
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.int32, [None, max_text_len])
y = tf.placeholder(tf.int32, [None])

x_onehot = tf.one_hot(x, alphabet_size)
y_onehot = tf.one_hot(y, class_num)

conv_x0 = tf.reshape(x_onehot, [-1, max_text_len, 1, alphabet_size])
# batch x 1014 x 1 x 69
conv_w1 = weight_init([7, 1, alphabet_size, 256])
conv_b1 = weight_init([256])
conv_x11 = tf.nn.conv2d(conv_x0, conv_w1, strides=[1, 1, 1, 1], padding="VALID")
conv_x12 = tf.nn.relu(conv_x11 + conv_b1)
conv_x13 = tf.nn.max_pool(conv_x12, ksize=[1,3,1,1], strides=[1,3,1,1], padding="VALID")
# batch x 336 x 1 x 256
conv_w2 = weight_init([7, 1, 256, 256])
conv_b2 = weight_init([256])
conv_x21 = tf.nn.conv2d(conv_x13, conv_w2, strides=[1, 1, 1, 1], padding="VALID")
conv_x22 = tf.nn.relu(conv_x21 + conv_b2)
conv_x23 = tf.nn.max_pool(conv_x22, ksize=[1,3,1,1], strides=[1,3,1,1], padding="VALID")
# batch x 110 x 1 x 256
conv_w3 = weight_init([3, 1, 256, 256])
conv_b3 = weight_init([256])
conv_x31 = tf.nn.conv2d(conv_x23, conv_w3, strides=[1, 1, 1, 1], padding="VALID")
conv_x32 = tf.nn.relu(conv_x31 + conv_b3)
# batch x 108 x 1 x 256
conv_w4 = weight_init([3, 1, 256, 256])
conv_b4 = weight_init([256])
conv_x41 = tf.nn.conv2d(conv_x32, conv_w4, strides=[1, 1, 1, 1], padding="VALID")
conv_x42 = tf.nn.relu(conv_x41 + conv_b4)
# batch x 106 x 1 x 256

# fully connection layer
x0 = tf.reshape(conv_x42, [-1, 106 * 1 * 256])

w1 = weight_init([106 * 1 * 256, 1024])
b1 = weight_init([1024])
x1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x0, w1) + b1), keep_prob)

w2 = weight_init([1024, 1024])
b2 = weight_init([1024])
x2 = tf.nn.dropout(tf.nn.relu(tf.matmul(x1, w2) + b2), keep_prob)

w3 = weight_init([1024, class_num])
b3 = weight_init([class_num])
y_predict = tf.nn.softmax(tf.matmul(x2, w3) + b3)

is_equal = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(y_predict, 1))
correct_count = tf.reduce_sum(tf.cast(is_equal, tf.int32))

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_onehot * tf.log(y_predict), reduction_indices=[1]))
train_step = tf.train.MomentumOptimizer(1e-2, 0.9).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# iterator
for epoch in range(epoch_count):
    random.shuffle(train_set)

    # train
    for batch_id in range( len(train_set) / batch_size ):
        start = batch_id * batch_size
        end = start + batch_size

        batch_x, batch_y = [], []
        for i in range(start, end):
            label, feature = train_set[i]
            batch_x.append(feature)
            batch_y.append(label)

        train_step.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

    # pridict
    total_correct_count = 0
    total_count = 0
    for batch_id in range( len(test_set) / batch_size ):
        start = batch_id * batch_size
        end = start + batch_size

        batch_x, batch_y = [], []
        for i in range(start, end):
            label, feature = test_set[i]
            batch_x.append(feature)
            batch_y.append(label)

        current_correct_count = correct_count.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_correct_count += current_correct_count
        total_count += batch_size
    accuracy = float(total_correct_count) / total_count
    print("epoch %d\taccuary %.4f"%(epoch+1, accuracy))



