#
# WORD VANILLA
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


def word_rnn_model_vanilla(x):
    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits, word_list


def data_read_words():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)

    return x_train, y_train, x_test, y_test, no_words


def main():
    global n_words

    x_train, y_train, x_test, y_test, n_words = data_read_words()

    print(len(x_train))
    print(len(x_test))

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    # graph for without dropouts
    logits_vanilla, word_ = word_rnn_model_vanilla(x)

    # Optimizer
    entropy_vanilla = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL),
                                                                 logits=logits_vanilla)
    loss_vanilla = tf.reduce_mean(entropy_vanilla)
    # Train without dropout
    train_op_vanilla = tf.train.AdamOptimizer(lr).minimize(loss_vanilla)
    # Accuracy without dropout
    accuracy_vanilla = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_vanilla, axis=1), y_), tf.float64))

    N = len(x_train)
    idx = np.arange(N)
    with tf.Session() as sess:
        test_acc_vanilla = []
        train_cost_vanilla = []
        sess.run(tf.global_variables_initializer())

        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op_vanilla.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})

            test_acc_vanilla.append(accuracy_vanilla.eval(feed_dict={x: x_test, y_: y_test}))
            train_cost_vanilla.append(loss_vanilla.eval(feed_dict={x: x_train, y_: y_train}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc_vanilla[e]))
            # print('Test accuracy: %g' % test_acc_vanilla[e])
            # print('Training cost: %g' % train_cost_vanilla[e])

    plt.figure(1)
    plt.plot(range(no_epochs), train_cost_vanilla, label='BasicRNN', color='Green')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Training Cost')
    plt.title('WORD VANILLA')

    plt.figure(2)
    plt.plot(range(no_epochs), test_acc_vanilla, label='BasicRNN', color='Green')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('WORD VANILLA')

    plt.show()


if __name__ == '__main__':
    main()