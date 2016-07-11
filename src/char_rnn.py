import tensorflow as tf
import numpy as np
from load import load
from collections import defaultdict
import random

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
max_text_len = 1014

#class_num, train_set, test_set = load("AG")
class_num, train_set, test_set = load("DBP")
train_set_group = defaultdict(list)
label_group = defaultdict(int)
for label, feature in train_set:
    train_set_group[label].append(feature)
    label_group[label] += 1

def get_batch(batch_size = 128):
    batch_x, batch_y = [], []
    for i in range(batch_size):
        label = random.randint(0, class_num-1)
        index = random.randint(0, label_group[label]-1)
        batch_x.append(train_set_group[label][index])
        batch_y.append(label)
    return batch_x, batch_y

# char rnn
hidden_size = 20

def weight_init(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.02, dtype=tf.float32)
    return tf.Variable(initial)

x = tf.placeholder(tf.int32, [None, max_text_len])
y = tf.placeholder(tf.int32, [None])
x_onehot = tf.one_hot(x, alphabet_size)
y_onehot = tf.one_hot(y, class_num)

input_list = tf.unpack(x_onehot, axis=1)
cell = tf.nn.rnn_cell.GRUCell(hidden_size)
output_list, state = tf.nn.rnn(cell, input_list, dtype=tf.float32)
w = weight_init([hidden_size, class_num])
b = weight_init([class_num])
y_predict = tf.nn.softmax(tf.matmul(state, w) + b)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_onehot * tf.log(y_predict), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# iterator
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(50000):
    batch_x, batch_y = get_batch()
    if i%1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
        print("step %d\ttraining accuracy %.4f"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch_x, y: batch_y})

total_accuracy = 0
for t in test_set:
    current_accuracy = accuracy.eval(feed_dict={x: [t[1]], y: [t[0]]})
    total_accuracy += current_accuracy
total_accuracy /= len(test_set)
print("test accuracy %.4f"%total_accuracy)


