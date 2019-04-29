#
# Project 2, Part B, Question 3
#

import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt
import time

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
batch_size = 128

no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_rnn_model(x):

    byte_vectors = tf.one_hot(x, 256)
    byte_list = tf.unstack(byte_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits

# Function to read data in
def read_data_chars():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            y_train.append(int(row[0]))

    with open('test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[1])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)

    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    x_test = np.array(list(char_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values

    return x_train, y_train, x_test, y_test

def main():

    x_train, y_train, x_test, y_test = read_data_chars()

    print(len(x_train))
    print(len(x_test))

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    logits = char_rnn_model(x)

    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits)
    loss = tf.reduce_mean(entropy)

    minimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = minimizer.compute_gradients(loss)

    # Gradient clipping
    grad_clipping = tf.constant(2.0, name="grad_clipping")
    clipped_grads_and_vars = []
    for grad, var in grads_and_vars:
        clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
        clipped_grads_and_vars.append((clipped_grad, var))

    # Gradient updates
    train_op = minimizer.apply_gradients(clipped_grads_and_vars)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))

    N = len(x_train)
    idx = np.arange(N)
    with tf.Session() as sess:
        test_acc = []
        train_cost = []

        sess.run(tf.global_variables_initializer())

        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]
            epoch_time = 0.0

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                #start_time = time.time()
                train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
                end_time = time.time()
                #epoch_time += end_time - start_time

            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
            train_cost.append(loss.eval(feed_dict={x: x_train, y_: y_train}))
            #total_time += epoch_time / float(N // batch_size)


            print('iteration: %d Test accuracy: %g' % (e,test_acc[e]))
            #print('Test accuracy: %g' % test_acc[e])
            #print('Training cost: %g' % train_cost[e])

        plt.figure(1)
        plt.plot(range(no_epochs), train_cost, label='Gradient Clipping')
        plt.xlabel('Epochs')
        plt.ylabel('Training Cost')
        plt.title('Training Error vs Epochs')

        plt.figure(2)
        plt.plot(range(no_epochs), test_acc, label='Adam Optimizer')
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs Epochs')

        plt.show()

if __name__ == '__main__':
    main()