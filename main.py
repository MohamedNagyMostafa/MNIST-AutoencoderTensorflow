import numpy as n
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

img = mnist.train.images[2]
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')

learning_rate = 0.001

with tf.device('/gpu:0'):
	# Input and target placeholders
	inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1))
	targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1))
    ### Encoder
	conv1 = tf.layers.conv2d(inputs_, 16, (3,3), padding='same', activation=tf.nn.relu)
    # Now 28x28x16
	maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), 2)
    # Now 14x14x16
	conv2 = tf.layers.conv2d(maxpool1, 8, (3,3), padding='same', activation=tf.nn.relu)
    # Now 14x14x8
	maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), 2)
    # Now 7x7x8
	conv3 = tf.layers.conv2d(maxpool2, 8, (3,3), padding='same', activation=tf.nn.relu)
    # Now 7x7x8
	encoded = tf.layers.max_pooling2d(conv2, (2,2), 2)
    # Now 4x4x8

    ### Decoder
	upsample1 = tf.image.resize_nearest_neighbor(encoded, (7,7))
    # Now 7x7x8
	conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
    # Now 7x7x8
	upsample2 = tf.image.resize_nearest_neighbor(conv4, (14, 14))
    # Now 14x14x8
	conv5 = tf.layers.conv2d(upsample2, 8, (3,3), padding='same', activation=tf.nn.relu)
    # Now 14x14x8
	upsample3 = tf.image.resize_nearest_neighbor(conv4, (28,28))
    # Now 28x28x8
	conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)
    # Now 28x28x16

	logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=tf.nn.relu)
    #Now 28x28x1

    # Pass logits through sigmoid to get reconstructed image
	decoded = tf.nn.sigmoid(logits)

    # Pass logits through sigmoid and calculate the cross-entropy loss
	loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets_)

    # Get cost and define the optimizer
	cost = tf.reduce_mean(loss)
	opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement = True))

epochs = 5
batch_size = 68
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs = batch[0].reshape((-1, 28, 28, 1))
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                         targets_: imgs})

        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[:10]
reconstructed = sess.run(decoded, feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1))})

for images, row in zip([in_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


fig.tight_layout(pad=0.1)