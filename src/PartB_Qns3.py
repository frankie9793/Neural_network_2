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


def char_rnn_model_dropout(x, keep_prob):
    byte_vectors = tf.one_hot(x, 256)
    byte_list = tf.unstack(byte_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, reuse=True)
    _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

    encoding = tf.nn.dropout(encoding, keep_prob)

    logits_dropout = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits_dropout


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
    keep_prob = tf.placeholder(tf.float32)

    # Build the graph without dropouts
    logits = char_rnn_model(x)
    # Build the graph with dropouts
    logits_dropout = char_rnn_model_dropout(x, keep_prob)

    # Optimizer without dropouts
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits)
    loss = tf.reduce_mean(entropy)

    # Optimizer with dropouts
    entropy_dropout = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL),
                                                                 logits=logits_dropout)
    loss_dropout = tf.reduce_mean(entropy_dropout)

    # train without dropouts
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    # train with dropouts
    train_op_dropout = tf.train.AdamOptimizer(lr).minimize(loss_dropout)

    # Accuracy without dropouts
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))
    # Accuracy with dropouts
    accuracy_dropouts = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_dropout, axis=1), y_), tf.float64))

    N = len(x_train)
    idx = np.arange(N)
    with tf.Session() as sess:
        test_acc = []
        train_cost = []
        total_epoch_time = 0.0
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})

            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
            train_cost.append(loss.eval(feed_dict={x: x_train, y_: y_train}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc[e]))
            # print('Test accuracy: %g' % test_acc[e])
            # print('Training cost: %g' % train_cost[e])

        end_time = time.time()
        total_epoch_time += end_time - start_time
        time_per_epoch = (total_epoch_time / float(no_epochs)) * 1000

        print("Runtime is %g" % time_per_epoch)

        plt.figure(1)
        plt.plot(range(no_epochs), train_cost, label='RNN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Cost')
        plt.legend()
        plt.title('RNN w/o dropout')

        plt.figure(2)
        plt.plot(range(no_epochs), test_acc, label='RNN')
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend
        plt.title('RNN w dropout')

        plt.show()

        print("Training with dropouts")
        print("===============================================================")
        test_acc_dropout = []
        train_cost_dropout = []
        total_epoch_time_dropout = 0.0
        sess.run(tf.global_variables_initializer())

        start_time_dropout = time.time()
        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op_dropout.run(feed_dict={x: x_train[start:end], y_: y_train[start:end], keep_prob: 0.5})

            test_acc_dropout.append(accuracy_dropouts.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))
            train_cost_dropout.append(loss_dropout.eval(feed_dict={x: x_train, y_: y_train, keep_prob: 1.0}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc_dropout[e]))
            # print('Test accuracy: %g' % test_acc_dropout[e])
            # print('Training cost: %g' % train_cost_dropout[e])

        end_time_dropout = time.time()
        total_epoch_time_dropout += end_time_dropout - start_time_dropout
        time_per_epoch = (total_epoch_time_dropout / float(no_epochs)) * 1000

        print("Runtime is %g" % time_per_epoch)

        plt.figure(1)
        plt.plot(range(no_epochs), train_cost_dropout, label='RNN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Cost')
        plt.legend()
        plt.title('RNN w dropout')

        plt.figure(2)
        plt.plot(range(no_epochs), test_acc_dropout, label='RNN')
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.title('RNN w dropout')

        plt.show()


if __name__ == '__main__':
    main()