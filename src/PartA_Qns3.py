#
# Project 2, Part A, Question 3
#


import tensorflow as tf
import numpy as np
import pylab as plt
import pickle

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 1000
batch_size = 128
FEATURE_SIZE1 = 80
FEATURE_SIZE2 = 90
momentum = 0.1


def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1

    return data, labels_


def cnn_withdropout(images):
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # First convolutional layer - maps 3 RGB image to 50 feature maps
    W_conv1 = tf.Variable(
        tf.truncated_normal([9, 9, NUM_CHANNELS, FEATURE_SIZE1], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
        name='weights_1')
    b_conv1 = tf.Variable(tf.zeros([FEATURE_SIZE1]), name='biases_1')
    u_conv1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
    h_conv1 = tf.nn.relu(u_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Second convolutional layer - maps 50 feature maps to 60
    W_conv2 = tf.Variable(
        tf.truncated_normal([5, 5, FEATURE_SIZE1, FEATURE_SIZE2], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
        name='weights_2')
    b_conv2 = tf.Variable(tf.zeros([FEATURE_SIZE2]), name='biases_2')
    u_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2
    h_conv2 = tf.nn.relu(u_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Fully connected layer 1 -- after 2 rounds of downsampling, our 3x32x32 image
    # is down to
    dim = h_pool2.get_shape()[1].value * h_pool2.get_shape()[2].value * h_pool2.get_shape()[3].value
    W_fc1 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0 / np.sqrt(dim)), name='weights_3')
    b_fc1 = tf.Variable(tf.zeros([300]), name='biases_3')

    h_pool2_flat = tf.reshape(h_pool2, [-1, dim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h1_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Softmax
    W_fc2 = tf.Variable(tf.truncated_normal([300, 10], stddev=1.0 / np.sqrt(dim)), name='weights_4')
    b_fc2 = tf.Variable(tf.zeros([10]), name='biases_4')

    y_conv = tf.matmul(h1_fc1_drop, W_fc2) + b_fc2

    return h_conv1, h_pool1, h_conv2, h_pool2, y_conv, keep_prob


def cnn(images):
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # First convolutional layer - maps 3 RGB image to 50 feature maps
    W_conv1 = tf.Variable(
        tf.truncated_normal([9, 9, NUM_CHANNELS, FEATURE_SIZE1], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
        name='weights_1')
    b_conv1 = tf.Variable(tf.zeros([FEATURE_SIZE1]), name='biases_1')
    u_conv1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
    h_conv1 = tf.nn.relu(u_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Second convolutional layer - maps 50 feature maps to 60
    W_conv2 = tf.Variable(
        tf.truncated_normal([5, 5, FEATURE_SIZE1, FEATURE_SIZE2], stddev=1.0 / np.sqrt(50 * 5 * 5)),
        name='weights_2')
    b_conv2 = tf.Variable(tf.zeros([FEATURE_SIZE2]), name='biases_2')
    u_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2
    h_conv2 = tf.nn.relu(u_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Fully connected layer 1 -- after 2 rounds of downsampling, our 3x32x32 image
    # is down to
    dim = h_pool2.get_shape()[1].value * h_pool2.get_shape()[2].value * h_pool2.get_shape()[3].value
    W_fc1 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0 / np.sqrt(dim)), name='weights_3')
    b_fc1 = tf.Variable(tf.zeros([300]), name='biases_3')

    h_pool2_flat = tf.reshape(h_pool2, [-1, dim])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Softmax
    W_fc2 = tf.Variable(tf.truncated_normal([300, 10], stddev=1.0 / np.sqrt(dim)), name='weights_4')
    b_fc2 = tf.Variable(tf.zeros([10]), name='biases_4')

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return h_conv1, h_pool1, h_conv2, h_pool2, y_conv


def main():
    # Read training data
    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)

    # Read testing data
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    # Scale the images
    trainX = (trainX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)
    testX = (testX - np.min(testX, axis=0)) / np.max(testX, axis=0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    h_conv1, h_pool1, h_conv2, h_pool2, logits = cnn(x)
    h_conv1_drop, h_pool1_drop, h_conv2_drop, h_pool2_drop, logits_drop, keep_prob = cnn_withdropout(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    cross_entropy_drop = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits_drop)

    loss = tf.reduce_mean(cross_entropy)
    loss_drop = tf.reduce_mean(cross_entropy_drop)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_step1 = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
    train_step2 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    train_step3 = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_step4 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_drop)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        correct_prediction_drop = tf.equal(tf.argmax(logits_drop, 1), tf.argmax(y_, 1))
        correct_prediction_drop = tf.cast(correct_prediction_drop, tf.float32)
        accuracy_drop = tf.reduce_mean(correct_prediction_drop)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        test_acc = []
        train_cost = []

        print('Training with Gradient Descent Optimizer without dropout')
        print('========================================')
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            train_cost.append(loss.eval(feed_dict={x: trainX, y_: trainY}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc[e]))
            # print('Test accuracy: %g' % test_acc[e])
            # print('Training cost: %g' % train_cost[e])

        # For training with momentum optimizer
        test_acc1 = []
        train_cost1 = []

        print('Training with Momentum Optimizer')
        print('================================')

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step1.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            test_acc1.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            train_cost1.append(loss.eval(feed_dict={x: trainX, y_: trainY}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc1[e]))
            # print('Test accuracy: %g' % test_acc1[e])
            # print('Training cost: %g' % train_cost1[e])

        # For training with RMSProp Algorithm

        test_acc2 = []
        train_cost2 = []

        print('Training with RMSProp Optimizer')
        print('================================')

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step2.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            test_acc2.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            train_cost2.append(loss.eval(feed_dict={x: trainX, y_: trainY}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc2[e]))
            # print('Test accuracy: %g' % test_acc2[e])
            # print('Training cost: %g' % train_cost2[e])

        # For Training with Adam Optimizer

        test_acc3 = []
        train_cost3 = []

        print('Training with Adam Optimizer')
        print('================================')

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step3.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            test_acc3.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            train_cost3.append(loss.eval(feed_dict={x: trainX, y_: trainY}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc3[e]))
            # print('Test accuracy: %g' % test_acc3[e])
            # print('Training cost: %g' % train_cost3[e])

        # For Training with Gradient Descent with dropouts
        test_acc4 = []
        train_cost4 = []

        print('Gradient Descent Optimizer with dropout')
        print('========================================')
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step4.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.5})

            test_acc4.append(accuracy_drop.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            train_cost4.append(loss_drop.eval(feed_dict={x: trainX, y_: trainY, keep_prob: 1.0}))

            print('iteration: %d Test accuracy: %g' % (e, test_acc4[e]))
            # print('Test accuracy: %g' % test_acc4[e])
            # print('Training cost: %g' % train_cost4[e])

        plt.figure(1, figsize=(30, 20))
        plt.plot(range(epochs), train_cost, label='GD w/o dropout')
        plt.plot(range(epochs), train_cost1, label='Momentum Optimizer')
        plt.plot(range(epochs), train_cost2, label='RMS Prop Optimizer')
        plt.plot(range(epochs), train_cost3, label='Adam Optimizer')
        plt.plot(range(epochs), train_cost4, label='GD w dropout')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Training Cost')
        plt.title('Training Error vs Epochs')
        # plt.savefig('./figures/PartA_Q3a_tvi.png')

        plt.figure(2, figsize=(30, 20))
        plt.plot(range(epochs), test_acc, label='GD w/o dropout')
        plt.plot(range(epochs), test_acc1, label='Momentum Optimizer')
        plt.plot(range(epochs), test_acc2, label='RMS Prop Optimizer')
        plt.plot(range(epochs), test_acc3, label='Adam Optimizer')
        plt.plot(range(epochs), test_acc4, label='GD w dropout')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs Epochs')
        # plt.savefig('./figures/PartA_Q3a_tvi.png')


if __name__ == '__main__':
    main()