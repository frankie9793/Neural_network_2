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

def char_rnn_model_2layers(x):

    byte_vectors = tf.one_hot(x, 256)
    byte_list = tf.unstack(byte_vectors, axis=1)

    cell1 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    cell2 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding[1], MAX_LABEL, activation=None)

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
    keep_prob = tf.placeholder(tf.float32)

    # Build the graph
    logits = char_rnn_model_2layers(x)

    # Optimizer
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits)
    loss = tf.reduce_mean(entropy)

    #train without dropouts
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    #Accuracy without dropouts
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

                train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})



            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
            train_cost.append(loss.eval(feed_dict={x: x_train, y_: y_train}))


            print('iteration: %d Test accuracy: %g' % (e,test_acc[e]))
            #print('Training cost: %g' % train_cost[e])

        plt.figure(1)
        plt.plot(range(no_epochs), train_cost, label='GRU')
        plt.xlabel('Epochs')
        plt.ylabel('Training Cost')
        plt.title('2Layers')

        plt.figure(2)
        plt.plot(range(no_epochs), test_acc, label='GRU')
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('2Layers')

        plt.show()


if __name__ == '__main__':
    main()