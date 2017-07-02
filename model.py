from enum import Enum

import tensorflow as tf

class Phase(Enum):
	Train = 0
	Validation = 1
	Predict = 2

class Model:
	def __init__(self, batch, label_batch, n_chars, hidden_sizes = [], phase = Phase.Predict, use_char_embeds = True) :
		batch_size = batch.shape[1]
		input_size = batch.shape[2]
		label_size = label_batch.shape[2]

		self._x = tf.placeholder(tf.int32, shape = [batch_size, input_size])

		if phase != Phase.Predict:
			self._y = tf.placeholder(tf.float32, shape = [batch_size, label_size])

		# build hidden layers
		layer = tf.reshape(tf.one_hot(self._x, n_chars), [batch_size, -1])
		for (i, hidden_size) in enumerate(hidden_sizes):
			W = tf.get_variable("W_hidden_%d" % i, shape = [layer.shape[1], hidden_size])
			b = tf.get_variable("b_hidden_%d" % i, shape = [hidden_size])
			hidden_outputs = tf.sigmoid(tf.matmul(layer, W) + b)
			layer = hidden_outputs
		hidden = layer
		
		# build output layer
		w = tf.get_variable("w", shape = [hidden.shape[1], label_size])
		b = tf.get_variable("b", shape = [1])
		
		# create loss function
		logits = tf.matmul(hidden, w) + b
		losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.cast(self._y, tf.float32), logits = logits)
		self._loss = tf.reduce_sum(losses)
		
		if phase == Phase.Train :
			self._train_op = tf.train.AdamOptimizer(0.004).minimize(self._loss) # converges much faster than gradient descent
		else :
			self._probs = tf.sigmoid(logits)
			self._labels = tf.cast(tf.round(self._probs), tf.int64)
			
			if phase == Phase.Validation :
				correct = tf.equal(tf.argmax(self._y), tf.argmax(self._labels)) # argmax because there are potentially multiple labels per token
				self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def loss(self):
		return self._loss

	@property
	def probs(self):
		return self._probs

	@property
	def train_op(self):
		return self._train_op

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y
