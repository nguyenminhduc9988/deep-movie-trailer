import os
import tensorflow as tf


# define input: w x h x d x color for C3D
#				w x h x color for facenet
#				1 x audio length for soundnet
# define output: 1 x length
# should return dictionary contain layer name and respective tensor for extraction of mid layer
def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def fusion_model(in_feature):
	# input: c3d output (4096) + soundnet output (4x1000) = 8096 dim
	# just normal NN with dropout, Resnet module, ...
	mu = 0
	sigma = 0.1
	keep_prob = 1.0
	in_normalize = tf.nn.l2_normalize(in_feature, 1)
	fc1_W = tf.Variable(tf.truncated_normal(shape=(8096, 15000), mean = mu, stddev = sigma))
	fc1_b = bias_variable([15000])
	fc1_pre   = tf.matmul(in_normalize, fc1_W) + fc1_b

	# SOLUTION: Activation.
	fc1   = tf.tanh(fc1_pre)

	fc1_out = None
	# keep probability number to be input in training
	# drop out

	fc1_out = tf.nn.dropout(fc1, keep_prob)


	# SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
	fc2_W  = tf.Variable(tf.truncated_normal(shape=(15000, 10000), mean = mu, stddev = sigma))
	fc2_b  = bias_variable([10000])
	fc2_pre = tf.matmul(fc1_out, fc2_W) + fc2_b

	# SOLUTION: Activation.
	fc2    = tf.tanh(fc2_pre)

	fc2_out = tf.nn.dropout(fc2, keep_prob)
	    
	# SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
	fc3_W  = tf.Variable(tf.truncated_normal(shape=(10000, 1000), mean = mu, stddev = sigma))
	fc3_b  = bias_variable([1000])
	fc3_pre = tf.matmul(fc2_out, fc3_W) + fc3_b

	fc3    = tf.tanh(fc3_pre)

	fc3_out = tf.nn.dropout(fc3, keep_prob)

	fc4_W  = tf.Variable(tf.truncated_normal(shape=(1000, 2), mean = mu, stddev = sigma))
	fc4_b  = bias_variable([2])
	fc4_pre = tf.matmul(fc3_out, fc4_W) + fc4_b

	fc4    = tf.tanh(fc4_pre)

	logits = fc4
	return logits

# def define_model(self, x, y):
#     mu = 0
#     sigma = 0.1
#     # resized image input 28x28x1 => 32x32x1
#     x_new = tf.map_fn(lambda x1: tf.image.pad_to_bounding_box(x1, 2, 2, 32, 32), x)
    
#     # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
#     conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
#     conv1_b = bias_variable([6])
#     conv1_pre   = tf.nn.conv2d(x_new, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

#     # SOLUTION: Activation.
#     conv1_ac = tf.tanh(conv1_pre)

#     # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
#     conv1 = tf.nn.max_pool(conv1_ac, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

#     # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
#     conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
#     conv2_b = bias_variable([16])
#     conv2_pre   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

#     # SOLUTION: Activation.
#     conv2_ac = tf.tanh(conv2_pre)

#     # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
#     conv2 = tf.nn.max_pool(conv2_ac, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

#     # SOLUTION: Flatten. Input = 5x5x16. Output = 400.t multiple values for keyword argument 'LR
#     fc0   = flatten(conv2)

#     # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
#     fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
#     fc1_b = tf.Variable(tf.zeros(120))
#     fc1_pre   = tf.matmul(fc0, fc1_W) + fc1_b

#     # SOLUTION: Activation.
#     fc1   = tf.tanh(fc1_pre)
    
#     fc1_out = None
#     # keep probability number to be input in training
#     # drop out
#     if (self.IsDropOut):
#         fc1_out = tf.nn.dropout(fc1, keep_prob)
#     else:
#         fc1_out = fc1

#     # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
#     fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
#     fc2_b  = bias_variable([84])
#     fc2_pre = tf.matmul(fc1_out, fc2_W) + fc2_b

#     # SOLUTION: Activation.
#     fc2    = tf.tanh(fc2_pre)
    
#     fc2_out = None
#     if (self.IsDropOut):
#         fc2_out = tf.nn.dropout(fc2, keep_prob)
#     else:
#         fc2_out = fc2
        
#     # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
#     fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
#     fc3_b  = bias_variable([10])
#     fc3_pre = tf.matmul(fc2_out, fc3_W) + fc3_b

#     logits = fc3_pre
#     return logits