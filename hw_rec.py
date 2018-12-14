import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time

'''old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
'''
'''from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
'''
outputs_dir = 'outputs'
'Size of flattened feature vector is 50x50'
box_axis_size = 50
n_classes = 62


print('Loading merged data ...')
X_train = np.load(os.path.join('X_train' + str(box_axis_size) + 'x' + str(box_axis_size)) + '.npy')
X_test = np.load(os.path.join('X_test' + str(box_axis_size) + 'x' + str(box_axis_size)) + '.npy')
Y_train = np.load(os.path.join('Y_train' + str(box_axis_size) + 'x' + str(box_axis_size)) + '.npy')
Y_test = np.load(os.path.join('Y_test' + str(box_axis_size) + 'x' + str(box_axis_size)) + '.npy')

'Merge for later shuffling'
train_data = [[X_train[tr_sample_idx], Y_train[tr_sample_idx]] for tr_sample_idx in range(len(X_train))]
test_data = [[X_test[test_sample_idx], Y_test[test_sample_idx]] for test_sample_idx in range(len(X_test))]

x1=[train_data[i][0] for i in range(0,len(train_data),2)]
y1=[train_data[i][1] for i in range(0,len(train_data),2)]

x2=[test_data[i][0] for i in range(0,len(test_data),2)]
y2=[test_data[i][1] for i in range(0,len(test_data),2)]
print(x2)

'PLOT TRAIN DATA'

for img_idx in range(len(X_train)):
	img = X_train[img_idx]
	img = np.reshape(img, newshape=[box_axis_size, box_axis_size])

	plt.imshow(img, cmap='gray')
	plt.show()

	label = Y_train[img_idx][1]



'''def next_batch(data, batch_size):

	np.random.shuffle(data)

	random_int_index = random.randint(0, len(data) - batch_size)

	mini_batch_data = data[random_int_index: random_int_index + batch_size]
	return mini_batch_data
#i for i in range(0,len(mini_batch_data),2):
#mini_batch_data[j for j in range(1,len(mini_batch_data),2)]
	batch_xs = []
	batch_ys = []
	for smpl_idx in range(len(mini_batch_data)):
		batch_xs.append(mini_batch_data[smpl_idx][0])
		batch_ys.append(mini_batch_data[smpl_idx][1])
		print(len(batch_ys[6]))
	return batch_xs, batch_ys
'''

def conv2d(x, weights):
	return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
	#						Size of the window	 movement of the window
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_neural_net(x, keep_prob, box_axis_size, n_classes, filter_size=5):

	stddev = np.sqrt(2) / np.sqrt(box_axis_size * box_axis_size)

	# Store layers weight & bias
	print("Stddev")
	print(stddev)    
	weights = {
		'wc1': tf.Variable(tf.random_normal([filter_size, filter_size, 1, 32], stddev=stddev), name='wc1'), # 3x3 conv, 1 input, 32 outputs
		'wc2': tf.Variable(tf.random_normal([filter_size, filter_size, 32, 64], stddev=stddev), name='wc2'), # 3x3 conv, 32 inputs, 64 outputs
		'wc3': tf.Variable(tf.random_normal([filter_size, filter_size, 64, 128], stddev=stddev), name='wc3'), # 3x3 conv, 32 inputs, 64 outputs

		'wd1': tf.Variable(tf.random_normal([7*7*128, 600], stddev=stddev), name='wd1'), # fully connected, 
		'out': tf.Variable(tf.random_normal([600, n_classes], stddev=stddev), name='out') # 1024 inputs, 2*10 output
	}

	biases = {
		'bc1': tf.Variable(tf.random_normal([32], stddev=stddev), name='bc1'),
		'bc2': tf.Variable(tf.random_normal([64], stddev=stddev), name='bc2'),
		'bc3': tf.Variable(tf.random_normal([128], stddev=stddev), name='bc3'),

		'bd1': tf.Variable(tf.random_normal([600], stddev=stddev), name='bd1'),
		'out': tf.Variable(tf.random_normal([n_classes], stddev=stddev), name='out')
	}

	x = tf.reshape(x, shape=[-1, box_axis_size, box_axis_size, 1])



	conv1 = tf.nn.relu(conv2d(x, weights=weights['wc1']) + biases['bc1'])
	conv1 = max_pool(conv1)
	conv2 = tf.nn.relu(conv2d(conv1, weights=weights['wc2']) + biases['bc2'])
	conv2 = max_pool(conv2)

	conv3 = tf.nn.relu(conv2d(conv2, weights=weights['wc3']) + biases['bc3'])
	conv3 = max_pool(conv3)

	fc = tf.reshape(conv3, shape=[-1, 7*7*128])
	fc = tf.nn.relu(tf.matmul(fc, weights['wd1']) + biases['bd1'])

	fc = tf.nn.dropout(fc, keep_prob)

	return tf.matmul(fc, weights['out']) + biases['out']



'HYPERPARAMS'
batch_size = 500
learning_rate_alpha = 0.0001
epoches = 600
dropout = 0.8
display_step = 5
filter_size = 3
x = tf.placeholder(tf.float32, [None, box_axis_size*box_axis_size])
y = tf.placeholder(tf.float32, [None, 62])
keep_prob = tf.placeholder(tf.float32)
print(keep_prob)


'Build a model'
pred = conv_neural_net(x, keep_prob, box_axis_size=box_axis_size, n_classes=n_classes, filter_size=filter_size)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_alpha).minimize(loss)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

losses = list()
accuracies = list()
 

saver = tf.train.Saver()
'Launch the graph'
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	start_epoch = time.time()

	for iteration in range(epoches):
		#batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		batch_xs, batch_ys = x1,y1
		c = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
		epoch_loss = c

		print('Epoch', iteration, 'completed out of', epoches, 'loss:', epoch_loss)

		if iteration % display_step == 0:
			losses.append(epoch_loss)

			# Calculate batch accuracy
			# batch_test_xs, batch_test_ys = mnist.train.next_batch(batch_size)
			batch_test_xs, batch_test_ys = x2,y2

			acc = sess.run(accuracy, feed_dict={x: batch_test_xs, y: batch_test_ys, keep_prob: 1.})
			accuracies.append(acc)
			print('Currect Accuracy', acc)

	end_epoch = time.time()

	print('Training took', end_epoch - start_epoch, 'seconds.')
