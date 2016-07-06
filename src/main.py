import tensorflow as tf
import numpy as np
from load import load

lphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
max_text_len = 1014
class_num = 4

train_set, test_set = load("AG")

# char cnn
x = tf.placeholder(tf.int32, [None, alphabet_size, max_text_len])
y = tf.placeholder(tf.int32, [None, class_num])



