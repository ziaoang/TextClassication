import tensorflow as tf
import numpy as np
from load import load
from collections import defaultdict
import random

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
max_text_len = 1014

class_num, train_set, test_set = load("AG")
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
x_image = tf.reshape(x_onehot, [-1, max_text_len, alphabet_size, 1])
# batch x 1014 x alphabet_size x 1
w1 = weight_init([7, alphabet_size, 1, 256])
b1 = weight_init([256])
x_11 = tf.nn.conv2d(x_image, w1, strides=[1, 1, 1, 1], padding="VALID")
x_12 = tf.nn.relu(x_11 + b1)
x_13 = tf.nn.max_pool(x_12, ksize=[1,3,1,1], strides=[1,3,1,1], padding="VALID")
# batch x 336 x 1 x 256
w2 = weight_init([7, 1, 256, 256])
b2 = weight_init([256])
x_21 = tf.nn.conv2d(x_13, w2, strides=[1, 1, 1, 1], padding="VALID")
x_22 = tf.nn.relu(x_21 + b2)
x_23 = tf.nn.max_pool(x_22, ksize=[1,3,1,1], strides=[1,3,1,1], padding="VALID")
# batch x 110 x 1 x 256
w3 = weight_init([3, 1, 256, 256])
b3 = weight_init([256])
x_31 = tf.nn.conv2d(x_23, w3, strides=[1, 1, 1, 1], padding="VALID")
x_32 = tf.nn.relu(x_31 + b3)
# batch x 108 x 1 x 256
w4 = weight_init([3, 1, 256, 256])
b4 = weight_init([256])
x_41 = tf.nn.conv2d(x_32, w4, strides=[1, 1, 1, 1], padding="VALID")
x_42 = tf.nn.relu(x_41 + b4)
# batch x 106 x 1 x 256
w5 = weight_init([3, 1, 256, 256])
b5 = weight_init([256])
x_51 = tf.nn.conv2d(x_42, w5, strides=[1, 1, 1, 1], padding="VALID")
x_52 = tf.nn.relu(x_51 + b5)
# batch x 104 x 1 x 256
w6 = weight_init([3, 1, 256, 256])
b6 = weight_init([256])
x_61 = tf.nn.conv2d(x_52, w6, strides=[1, 1, 1, 1], padding="VALID")
x_62 = tf.nn.relu(x_61 + b6)
x_63 = tf.nn.max_pool(x_62, ksize=[1,3,1,1], strides=[1,3,1,1], padding="VALID")
# batch x 34 x 1 x 256
x_63_flat = tf.reshape(x_63, [-1, 34*1*256])
# batch x 8704
w7 = weight_init([34*1*256, 1024])
b7 = weight_init([1024])
x_71 = tf.matmul(x_63_flat, w7) + b7
x_72 = tf.nn.relu(x_71)
x_73 = tf.nn.dropout(x_72, keep_prob)
# batch x 1024
w8 = weight_init([1024, 1024])
b8 = weight_init([1024])
x_81 = tf.matmul(x_73, w8) + b8
x_82 = tf.nn.relu(x_81)
x_83 = tf.nn.dropout(x_82, keep_prob)
# batch x 1024
w9 = weight_init([1024, class_num])
b9 = weight_init([class_num])
x_91 = tf.matmul(x_83, w9) + b9
x_92 = tf.nn.softmax(x_91)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_onehot * tf.log(x_92), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(x_92, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# iterator
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(50000):
    batch_x, batch_y = get_batch()
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch_x, y: batch_y, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: [t[1] for t in test_set],
    y: [t[0] for t in test_set],
    keep_prob: 1.0}))



