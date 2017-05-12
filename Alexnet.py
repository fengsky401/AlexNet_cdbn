
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib import layers

from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import global_define as gd
import scipy.io as sio
from tensorflow.python.ops import nn

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

#log_name=str(FLAGS.log_dir)+'zooscan_224_224_20_'+'layer5'+'.txt'
load_fn='/home/scw4750/Liuhongkun/tfrecord/zooscan/Alex_cdbn/data_record/experiment2/Layer5/log2017022714343244326/parameters_layer4_epoch_400.mat'
load_data=sio.loadmat(load_fn)

W1=load_data['W1']
print(W1.shape)
#a1=load_data['a1']
b1=load_data['b1']
W2=load_data['W2']
b2=load_data['b2']
W3=load_data['W3']
b3=load_data['b3']
W4=load_data['W4']
b4=load_data['b4']
W5=load_data['W5']
b5=load_data['b5']


def weight_variable(name,shape,initial_value):
	
	return tf.get_variable(name=name,shape=shape,initializer=tf.truncated_normal_initializer(stddev=initial_value))

def weight_decay(name, shape, stddev, wd):
	var=tf.get_variable(name=name,shape=shape,initializer=tf.truncated_normal_initializer(stddev=stddev))
	if wd is not None:
		weight_decay=tf.mul(tf.nn.l2_loss(var),wd,name='weight_loss')
		tf.add_to_collection('losses',weight_decay)
	#print(var)
	return var

def bias_variable(name,shape,initial_value):
	
	return tf.get_variable(name=name,shape=shape,initializer=tf.constant_initializer(initial_value))

def conv2d(x,W,stride,padding):
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)


def max_pool(x,ksize,stride,padding):
	return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


def alexnet_v2_arg_scope(weight_decay=0.0005):
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      activation_fn=nn_ops.relu,
      biases_initializer=init_ops.constant_initializer(0.1),
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope([layers.conv2d], padding='SAME'):
      with arg_scope([layers_lib.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc

def inference(image_input):
	#scope=alexnet_v2_arg_scope()
	num_classes=20
	is_training=True
	dropout_keep_prob=0.5
	spatial_squeeze=True
	scope='alexnet_v2'
#with slim.arg_scope(alexnet_v2_arg_scope()):
	with variable_scope.variable_scope(scope, 'alexnet_v2', [image_input]) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
		with arg_scope(
		[layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
		outputs_collections=[end_points_collection]) as here1:
		#print(a)
			net = layers.conv2d(
			  image_input, 64, [11, 11], 4, 
			  weights_initializer=tf.constant_initializer(W1),
			  biases_initializer=tf.constant_initializer(b1),
			   activation_fn=nn.sigmoid,padding='VALID', scope='conv1')
			net = layers_lib.max_pool2d(net, [3, 3], 2,scope='pool1')
			net = layers.conv2d(net, 192, [5, 5],
				weights_initializer=tf.constant_initializer(W2),
				biases_initializer=tf.constant_initializer(b2), 
				activation_fn=nn.sigmoid,scope='conv2')
			net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
			net = layers.conv2d(net, 384, [3, 3], 
				weights_initializer=tf.constant_initializer(W3),
				biases_initializer=tf.constant_initializer(b3),
				activation_fn=nn.sigmoid,scope='conv3')
			net = layers.conv2d(net, 384, [3, 3],
				weights_initializer=tf.constant_initializer(W4),
				biases_initializer=tf.constant_initializer(b4), 
				activation_fn=nn.sigmoid,scope='conv4')
			net = layers.conv2d(net, 256, [3, 3],
				weights_initializer=tf.constant_initializer(W5),
				biases_initializer=tf.constant_initializer(b5),
				activation_fn=nn.sigmoid, scope='conv5')
			net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')
		# print()
			#print(here1)
			# Use conv2d instead of fully_connected layers.
			with arg_scope(
			  [layers.conv2d],
			  weights_initializer=trunc_normal(0.005),
			  biases_initializer=init_ops.constant_initializer(0.1)) as here2:
				#print(here2)
				a=3
				net = layers.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
				net = layers_lib.dropout(
				    net, dropout_keep_prob, is_training=is_training, scope='dropout6')
				net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
				net = layers_lib.dropout(
				    net, dropout_keep_prob, is_training=is_training, scope='dropout7')
				net = layers.conv2d(
				    net,
				    num_classes, [1, 1],
				    activation_fn=None,
				    normalizer_fn=None,
				    biases_initializer=init_ops.zeros_initializer(),
				    scope='fc8')
				#print(a)

				# Convert end_points_collection into a end_point dict.
				end_points = utils.convert_collection_to_dict(end_points_collection)
				if spatial_squeeze:
					net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
					end_points[sc.name + '/fc8'] = net
				return net, end_points
	#	return logits

def loss(h_fc3,labels):
	print('labels:'+str(labels))
	batch_size=tf.size(labels)
	labels=tf.expand_dims(labels,1)
	indices=tf.expand_dims(tf.range(0,batch_size),1)
	# print('indices:'+str(indices))
	# print('labels:'+str(labels))
	#concated=tf.concat(1,[indices,labels])
	concated=tf.concat([indices,labels],1)
	onehot_labels=tf.sparse_to_dense(concated,tf.stack([batch_size,gd.NUM_CLASSES]),1.0,0.0)
	cross_entropy=slim.losses.softmax_cross_entropy(h_fc3,onehot_labels)
	loss=tf.reduce_mean(cross_entropy,name='xentropy_mean')
	tf.summary.scalar('xentropy_mean',loss)
	return loss


def training(loss,learning_rate):
	optimizer=tf.train.GradientDescentOptimizer(learning_rate)
	global_step=tf.Variable(0,name='global_step',trainable=False)
	train_op=optimizer.minimize(loss,global_step=global_step)
	return train_op

# def recall(logits,labels):
# 	batch_size=tf.size(labels)
# 	labels=tf.expand_dims(labels,1)
# 	indices=tf.expand_dims(tf.range(0,batch_size),1)
# 	# print('indices:'+str(indices))
# 	# print('labels:'+str(labels))
# 	#concated=tf.concat(1,[indices,labels])
# 	concated=tf.concat([indices,labels],1)
# 	onehot_labels=tf.sparse_to_dense(concated,tf.stack([batch_size,gd.NUM_CLASSES]),1.0,0.0)
# 	recall,_=tf.contrib.metrics.streaming_recall(logits, onehot_labels, 
# 		weights=None, metrics_collections=None, updates_collections=None, name=None)
# 	tf.summary.scalar('recall',recall)
# 	return recall




def evaluation(h_fc3,labels):
	correct=tf.nn.in_top_k(h_fc3,labels,1)
	tf.summary.scalar('evaluation',tf.reduce_sum(tf.cast(correct,tf.int32)))
	return tf.reduce_sum(tf.cast(correct,tf.int32))

def confusion_matrix(logits,labels):
	batch_size=tf.size(labels)
	labels=tf.expand_dims(labels,1)
	indices=tf.expand_dims(tf.range(0,batch_size),1)

	concated=tf.concat([indices,labels],1)
	onehot_labels=tf.sparse_to_dense(concated,tf.stack([batch_size,gd.NUM_CLASSES]),1.0,0.0)

	predicted = tf.round(tf.nn.sigmoid(logits))
	actual = labels

	predicted=tf.cast(predicted,tf.float32)
	actual=tf.cast(predicted,tf.float32)

	tp = tf.count_nonzero(predicted * actual)
	tn = tf.count_nonzero((predicted - 1) * (actual - 1))
	fp = tf.count_nonzero(predicted * (actual - 1))
	fn = tf.count_nonzero((predicted - 1) * actual)
	    
	# Calculate accuracy, precision, recall and F1 score.
	accuracy = (tp + tn) / (tp + fp + fn + tn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	fmeasure = (2 * precision * recall) / (precision + recall)

	# Add metrics to TensorBoard.    
	tf.summary.scalar('Accuracy', accuracy)
	tf.summary.scalar('Precision', precision)
	tf.summary.scalar('Recall', recall)
	tf.summary.scalar('f-measure', fmeasure)

	return recall

# def  accuracy(h_fc3,labels):
# 	batch_size=tf.size(labels)
# 	labels=tf.expand_dims(labels,1)
# 	indices=tf.expand_dims(tf.range(0,batch_size),1)
# 	# print('indices:'+str(indices))
# 	# print('labels:'+str(labels))
# 	#concated=tf.concat(1,[indices,labels])
# 	concated=tf.concat([indices,labels],1)
# 	onehot_labels=tf.sparse_to_dense(concated,tf.stack([batch_size,gd.NUM_CLASSES]),1.0,0.0)
# 	accuracy,_=tf.contrib.metrics.streaming_accuracy(h_fc3, onehot_labels,
# 	 weights=None, metrics_collections=None, updates_collections=None, name=None)
# 	tf.summary.scalar('accuracy',accuracy)
# 	return accuracy