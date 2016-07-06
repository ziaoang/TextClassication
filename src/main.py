import tensorflow as tf
import numpy as np
from load import load

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
max_text_len = 1014
class_num = 4

train_set, test_set = load("AG")

# char cnn
x = tf.placeholder(tf.int32, [None, max_text_len])
y = tf.placeholder(tf.int32, [None, 1])

x_onehot = tf.one_hot(x, alphabet_size)
y_onehot = tf.one_hot(y, class_num)

x_image = tf.reshape(x_onehot, [-1, max_text_len, alphabet_size, 1])

print(x.get_shape())
print(y.get_shape())
print(x_onehot.get_shape())
print(y_onehot.get_shape())

# 1 - layer
