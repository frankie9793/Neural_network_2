#
# Project 2, Part A, Question 1
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import pylab

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 1000
batch_size = 128


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


def cnn(images):
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # First convolutional layer - maps 3 RGB image to 50 feature maps
    W_conv1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)),
                          name='weights_1')
    b_conv1 = tf.Variable(tf.zeros([50]), name='biases_1')
    u_conv1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
    h_conv1 = tf.nn.relu(u_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Second convolutional layer - maps 50 feature maps to 60
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0 / np.sqrt(50 * 5 * 5)),
                          name='weights_2')
    b_conv2 = tf.Variable(tf.zeros([60]), name='biases_2')
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

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        test_acc = []
        train_cost = []
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            train_cost.append(loss.eval(feed_dict={x: trainX, y_: trainY}))

            if (e % 100 == 0):
                print('iteration: %d' % e)
                print('Test accuracy: %g' % test_acc[e])
                print('Training cost: %g' % train_cost[e])

        ind = np.random.randint(low=0, high=2000)
        X = testX[ind, :]
        ind2 = np.random.randint(low=0, high=2000)
        X2 = testX[ind2, :]

        # For the first image in the test batch
        h_conv_1, h_pool_1, = sess.run([h_conv1, h_pool1], {x: X.reshape(1, IMG_SIZE * IMG_SIZE * NUM_CHANNELS)})
        h_conv_2, h_pool_2, = sess.run([h_conv2, h_pool2], {x: X.reshape(1, IMG_SIZE * IMG_SIZE * NUM_CHANNELS)})

        # For the second image in the test batch
        h_conv_11, h_pool_11 = sess.run([h_conv1, h_pool1], {x: X2.reshape(1, IMG_SIZE * IMG_SIZE * NUM_CHANNELS)})
        h_conv_22, h_pool_22, = sess.run([h_conv2, h_pool2], {x: X2.reshape(1, IMG_SIZE * IMG_SIZE * NUM_CHANNELS)})

        # Accuracy graphs below
        plt.figure(1)
        plt.plot(range(epochs), train_cost, label='Gradient Descent')
        plt.xlabel('Epochs')
        plt.ylabel('Training Cost')
        plt.title('Training Error vs Epochs')
        # plt.savefig('./figures/PartA_Q3a_tvi.png')

        plt.figure(2)
        plt.plot(range(epochs), test_acc, label='Gradient Descent')
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs Epochs')
        # plt.savefig('./figures/PartA_Q3a_tvi.png')

        # For the first random image from the test batch
        # original image
        pylab.figure()
        X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
        plt.axis('off')
        plt.imshow(X_show)

        # and its feature maps
        pylab.figure()
        pylab.gray()
        h_conv1_ = np.array(h_conv_1)
        for i in range(50):
            pylab.subplot(5, 10, i + 1);
            pylab.axis('off');
            pylab.imshow(h_conv1_[0, :, :, i])
        # pylab.savefig('./figures/7.3_4.png')

        pylab.figure()
        pylab.gray()
        h_pool1_ = np.array(h_pool_1)
        for i in range(50):
            pylab.subplot(5, 10, i + 1);
            pylab.axis('off');
            pylab.imshow(h_pool1_[0, :, :, i])
        # pylab.savefig('./figures/7.3_5.png')

        pylab.figure()
        pylab.gray()
        h_conv2_ = np.array(h_conv_2)
        for i in range(60):
            pylab.subplot(6, 10, i + 1);
            pylab.axis('off');
            pylab.imshow(h_conv2_[0, :, :, i])
        # pylab.savefig('figures/7.3_6.png')

        pylab.figure()
        pylab.gray()
        h_pool2_ = np.array(h_pool_2)
        for i in range(60):
            pylab.subplot(6, 10, i + 1);
            pylab.axis('off');
            pylab.imshow(h_pool2_[0, :, :, i])
        # pylab.savefig('./figures/7.3_7.png')

        # For the second random image
        # For the first random image from the test batch
        # original image
        pylab.figure()
        X2_show = X2.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
        plt.axis('off')
        plt.imshow(X2_show)

        # and its feature maps
        pylab.figure()
        pylab.gray()
        h_conv11_ = np.array(h_conv_11)
        for i in range(50):
            pylab.subplot(5, 10, i + 1);
            pylab.axis('off');
            pylab.imshow(h_conv11_[0, :, :, i])
        # pylab.savefig('./figures/7.3_4.png')

        pylab.figure()
        pylab.gray()
        h_pool11_ = np.array(h_pool_11)
        for i in range(50):
            pylab.subplot(5, 10, i + 1);
            pylab.axis('off');
            pylab.imshow(h_pool11_[0, :, :, i])
        # pylab.savefig('./figures/7.3_5.png')

        pylab.figure()
        pylab.gray()
        h_conv22_ = np.array(h_conv_22)
        for i in range(60):
            pylab.subplot(6, 10, i + 1);
            pylab.axis('off');
            pylab.imshow(h_conv22_[0, :, :, i])
        # pylab.savefig('figures/7.3_6.png')

        pylab.figure()
        pylab.gray()
        h_pool22_ = np.array(h_pool_22)
        for i in range(60):
            pylab.subplot(6, 10, i + 1);
            pylab.axis('off');
            pylab.imshow(h_pool22_[0, :, :, i])
        # pylab.savefig('./figures/7.3_7.png')

        pylab.show()


if __name__ == '__main__':
    main()