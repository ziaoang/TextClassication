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
x = tf.placeholder(tf.int32, [None, max_text_len])
y = tf.placeholder(tf.int32, [None])

x_onehot = tf.one_hot(x, alphabet_size)
y_onehot = tf.one_hot(y, class_num)

hidden_size = 20
input_list = tf.unpack(x_onehot, axis=1)
cell = tf.nn.rnn_cell.GRUCell(hidden_size)
# state = cell.zero_state(...)
# outputs = []
# for input_ in inputs: # A length T list of inputs, each a tensor of shape [batch_size, input_size]
#     output, state = cell(input_, state)
#     outputs.append(output)
# return (outputs, state)
output_list, state = tf.nn.rnn(cell, input_list, dtype=tf.float32)
output_list_mean = tf.reduce_mean(output_list, reduction_indices=0)
w = weight_init([hidden_size, class_num])
b = weight_init([class_num])
y_predict = tf.nn.softmax(tf.matmul(output_list_mean, w) + b)

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

        train_step.run(feed_dict={x: batch_x, y: batch_y})

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

        current_correct_count = correct_count.eval(feed_dict={x: batch_x, y: batch_y})
        total_correct_count += current_correct_count
        total_count += batch_size
    accuracy = float(total_correct_count) / total_count
    print("epoch %d\taccuary %.4f"%(epoch+1, accuracy))



