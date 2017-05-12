from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np
import  re
import datetime
import utils
import tensorflow as tf
import PIL
from PIL import Image
import global_define as gd

from utils import tile_raster_images
import scipy.io as sio

def weight_variable(name,shape,initial_value):
	# initial=tf.truncated_normal(shape,stddev=0.1)
	# return tf.Variable(initial)
	return tf.get_variable(name=name,shape=shape,initializer=tf.random_normal_initializer(stddev=initial_value))


def bias_variable(name,shape,initial_value):
	# initial=tf.constant(0.1,shape=shape)
	# return tf.Variable(initial)
	return tf.get_variable(name=name,shape=shape,initializer=tf.constant_initializer(initial_value))

def conv2d(x,W,stride,padding):
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)


def max_pool(x,ksize,stride,padding):
	return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

input_size=27
chanl_input=96
#chanl_out
conv_size=5
chanl_output=256

W_conv1=tf.placeholder("float",[conv_size,conv_size,chanl_input,chanl_output])
a_conv1=tf.placeholder("float",[chanl_input])
b_conv1=tf.placeholder("float",[chanl_output])

W_inc1=tf.placeholder("float",[conv_size,conv_size,chanl_input,chanl_output])
a_inc1=tf.placeholder("float",[chanl_input])
b_inc1=tf.placeholder("float",[chanl_output])

W_extra1=tf.placeholder("float",[11,11,1,96])
#a_extra1=tf.placeholder("float",[1])
b_extra1=tf.placeholder("float",[96])


images_placeholder=tf.placeholder(tf.float32,shape=(gd.BATCH_SIZE,227*227))
images_extra=tf.reshape(images_placeholder,[-1,227,227,1])

h_conv1=1./(1+tf.exp(-conv2d(images_extra,W_extra1,4,'VALID')-b_extra1))
norm1=tf.nn.lrn(h_conv1,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm1')
h_pool1=max_pool(norm1,3,2,'VALID')


images=h_pool1
print(images)


pos_conv1_prob=1./(1+tf.exp(-conv2d(h_pool1,W_conv1,1,'VALID')-b_conv1))
pos_conv1_trans=tf.expand_dims(tf.reduce_mean(pos_conv1_prob,0),2)
images_mean=tf.reduce_mean(images,0)
images_trans=tf.expand_dims(tf.reshape(tf.transpose(tf.reshape(
	tf.reduce_mean(images,0),[-1,chanl_input])),[chanl_input,input_size,input_size]),3)

pos_prods_origin=conv2d(images_trans,pos_conv1_trans,1,'VALID')

pos_prods_trans=tf.transpose(pos_prods_origin,[1,2,0,3])
print('pos_prods_trans:'+str(pos_prods_trans))

pos_hid_act=tf.reduce_mean(pos_conv1_prob,0)

pos_vis_act=tf.reduce_mean(images,0)
#########################################################################3
# pos_hid_states=tf.to_float(tf.less_equal(tf.random_uniform(
# 				shape=tf.shape(pos_conv1_prob)),pos_conv1_prob))

#if pad_choose=="VALID":
pos_conv1_prob_same=1./(1+tf.exp(-conv2d(h_pool1,W_conv1,1,'SAME')-b_conv1))
pos_hid_states_same=tf.to_float(tf.less_equal(tf.random_uniform(
			shape=tf.shape(pos_conv1_prob_same)),pos_conv1_prob_same))
#else:


W_transpose=tf.matrix_transpose(tf.reverse(W_conv1,[True,True,False,False]))

print('pos_conv1_prob:'+str(pos_conv1_prob))
neg_data=1./(1+tf.exp(-conv2d(pos_hid_states_same,W_transpose,1,'SAME')-a_conv1))

#neg_data=1./(1+tf.exp(-conv2d_s1_valid(pos_hid_states_addpad,W_transpose)-a_conv1))
#neg_data=
print('neg_data:'+str(neg_data))
neg_hid_probs=1./(1+tf.exp(-conv2d(neg_data,W_conv1,1,'VALID')-b_conv1))

neg_data_trans=tf.expand_dims(tf.reshape(tf.transpose(tf.reshape(
	tf.reduce_mean(neg_data,0),[-1,chanl_input])),[chanl_input,input_size,input_size]),3)


neg_hid_probs_trans=tf.expand_dims(tf.reduce_mean(neg_hid_probs,0),2)

neg_prods_origin=conv2d(neg_data_trans,neg_hid_probs_trans,1,'VALID')

neg_prods_trans=tf.transpose(neg_prods_origin,[1,2,0,3])
print('neg_prods_trans'+str(neg_prods_trans))
neg_hid_act=tf.reduce_mean(neg_hid_probs,0)
neg_vis_act=tf.reduce_mean(neg_data,0)
