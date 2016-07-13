import tensorflow as tf
import numpy as np
from load import load
from collections import defaultdict
import random

epoch_count = 100
batch_size = 128

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
max_text_len = 1014

class_num, train_set, test_set = load("AG")
# class_num, train_set, test_set = load("DBP")

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
x0 = tf.reshape(x_onehot, [-1, max_text_len * alphabet_size])

w1 = weight_init([max_text_len * alphabet_size, 2048])
b1 = weight_init([2048])
x1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x0, w1) + b1), keep_prob)

w2 = weight_init([2048, 2048])
b2 = weight_init([2048])
x2 = tf.nn.dropout(tf.nn.relu(tf.matmul(x1, w2) + b2), keep_prob)

w3 = weight_init([2048, class_num])
b3 = weight_init([class_num])
x3 = tf.nn.softmax(tf.matmul(x2, w3) + b3)

is_equal = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(x3, 1))
correct_count = tf.reduce_sum(tf.cast(is_equal, tf.int32))

# loss function
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_onehot * tf.log(x3), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)

# iterator
random.seed(123456789)
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


