#
# CHARACTER LSTM
#

import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20
batch_size = 128

no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def char_rnn_model_lstm(x):
    byte_vectors = tf.one_hot(x, 256)
    byte_list = tf.unstack(byte_vectors, axis=1)

    cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, use_peepholes=True)

    # initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding[1], MAX_LABEL, activation=None)

    return logits


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

    logits_lstm = char_rnn_model_lstm(x)

    entropy_lstm = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits_lstm)
    loss_lstm = tf.reduce_mean(entropy_lstm)
    train_op_lstm = tf.train.AdamOptimizer(lr).minimize(loss_lstm)
    accuracy_lstm = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_lstm, axis=1), y_), tf.float64))

    N = len(x_train)
    idx = np.arange(N)
    with tf.Session() as sess:
        test_acc_lstm = []
        train_cost_lstm = []
        sess.run(tf.global_variables_initializer())

        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op_lstm.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})

            test_acc_lstm.append(accuracy_lstm.eval(feed_dict={x: x_test, y_: y_test}))
            train_cost_lstm.append(loss_lstm.eval(feed_dict={x: x_train, y_: y_train}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc_lstm[e]))
            # print('Test accuracy: %g' % test_acc_lstm[e])
            # print('Training cost: %g' % train_cost_lstm[e])

    plt.figure(1)
    plt.plot(range(no_epochs), train_cost_lstm, label='BasicLSTM', color='Red')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Training Cost')
    plt.title('CHARACTER LSTM')

    plt.figure(2)
    plt.plot(range(no_epochs), test_acc_lstm, label='BasicLSTM', color='Red')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Test Accuracy')
    plt.title('CHARACTER LSTM')

    plt.show()


if __name__ == '__main__':
    main()