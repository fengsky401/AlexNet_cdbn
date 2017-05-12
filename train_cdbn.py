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

np.set_printoptions(threshold='nan') 

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE='test.tfrecords'
weightcost=0.0000002

flags = tf.app.flags
FLAGS = flags.FLAGS
#FLAGS = None
time_value=re.sub(r'[^0-7]','',str(datetime.datetime.now()))
flags.DEFINE_string('tfrecord_dir', 
	'/media/scw4750/25a01ed5-a903-4298-87f2-a5836dcb6888/Liuhongkun/zooscan_2/shuffled/224_224/', 'Directory to put the training data.')
flags.DEFINE_string('filename', 'train.tfrecords', 'Directory to put the training data.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('num_epochs', None, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')

flags.DEFINE_integer('learning_rate', 0.1,'balabala')
flags.DEFINE_integer('max_steps', 50000,'balabala')
flags.DEFINE_string('model_dir','Modal/model'+str(time_value)+'/','balabala')
flags.DEFINE_string('tensorevents_dir','tensorboard_event/event_wth'+str(time_value)+'/','balabala')
flags.DEFINE_string('log_dir','Log_data/log'+str(time_value)+'/','balabala')
flags.DEFINE_string('pic_dir','Pic/Pictures_input'+str(time_value)+'/','balabala')
flags.DEFINE_string('Weight_dir','Weight/weight'+str(time_value)+'/','balabala')
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

if not os.path.exists(FLAGS.tensorevents_dir):
  os.makedirs(FLAGS.tensorevents_dir)

if not os.path.exists(FLAGS.model_dir):
	os.makedirs(FLAGS.model_dir)

if not os.path.exists(FLAGS.pic_dir):
	os.makedirs(FLAGS.pic_dir)

if not os.path.exists(FLAGS.Weight_dir):
	os.makedirs(FLAGS.Weight_dir)
def to_image(flat_tensor):
	if not isinstance(flat_tensor,np.ndarray):
		flat_array=tf.squeeze(flat_tensor).eval()
	else:
		flat_array=flat_tensor
	#print(flat_array.shape)
	if  len(flat_array.shape)==1:
		flat_length=flat_array.shape
		square_length=int(np.sqrt(flat_length))
		square_array=flat_array.reshape([square_length,square_length])
	elif len(flat_array.shape)==2:
		batch_size,flat_length=flat_array.shape
		square_length=int(np.sqrt(flat_length))
		square_array=flat_array.reshape([batch_size,square_length,square_length])
	elif len(flat_array.shape)==3:
		square_array=flat_array
	else :
		batch_size,square_width,square_height,depth=flat_array.shape
		# print(flat_array.shape)
		# print('size of the image input is %d,%d,%d,%d' %(batch_size,square_width,square_height,depth))
		square_array=flat_array[0,:,:,:]

	
	return square_array

def conv2d_s4_same(x,W):
    return tf.nn.conv2d(x,W,strides=[1,4,4,1],padding='SAME')
def conv2d_s4_valid(x,W):
	return tf.nn.conv2d(x,W,strides=[1,4,4,1],padding='VALID')

def conv2d_s16_valid(x,W):
    return tf.nn.conv2d(x,W,strides=[1,16,16,1],padding='VALID')

def weight_variable(name,shape):
	return tf.get_variable(name=name,shape=shape,initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name,shape):
	return tf.get_variable(name=name,shape=shape,initializer=tf.constant_initializer(0.1))

def conv2d_s1_valid(x,W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def conv2d_s1_same(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_3x3(x):
	return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


def conv2d(x,W,stride,padding):
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)


def max_pool(x,ksize,stride,padding):
	return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)


def read_and_decode(filename_queue):
	reader=tf.TFRecordReader()
	_,serialized_exampe=reader.read(filename_queue)
	features=tf.parse_single_example(serialized_exampe,
		features={
		'image_raw':tf.FixedLenFeature([],tf.string),
		'label':tf.FixedLenFeature([],tf.int64)
		})
	image=tf.decode_raw(features['image_raw'],tf.uint8)
	image.set_shape([gd.IMAGE_PIXELS])
	image=tf.cast(image,tf.float32)*(1./255)-0.5
	label=tf.cast(features['label'],tf.int32)
	return image,label

def do_eval(sess,eval_correct,log_name):
	true_count=0
	for step in xrange(FLAGS.batch_size):
		#print(sess.run(eval_correct))
		true_count+=sess.run(eval_correct)

	precision=float(true_count)/(FLAGS.batch_size*FLAGS.batch_size)
	# true_count=sess.run(eval_correct)
	# precision=float(true_count)/FLAGS.batch_size
	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (FLAGS.batch_size*FLAGS.batch_size, true_count, precision))
	logfile=open(log_name,'a')
	logfile.write('  Num examples: %d  Num correct: %d  Precision : %0.04f' %
            (FLAGS.batch_size, true_count, precision))
	logfile.close()
	return precision


def inputs(train,batch_size,num_epochs):
	if not num_epochs:num_epochs=None
	if train=='train':
		filename=os.path.join(FLAGS.tfrecord_dir,TRAIN_FILE)
	elif train=='validation':
		filename=os.path.join(FLAGS.tfrecord_dir,VALIDATION_FILE)
	else:
		filename=os.path.join(FLAGS.tfrecord_dir,TEST_FILE)

	with tf.name_scope('input'):
		filename_queue=tf.train.string_input_producer([filename],num_epochs=None)
		# example_list=[read_and_decode(filename_queue) for _ in range(10)]
		# images,sparse_labels=tf.train.shuffle_batch_join(
		# 	example_list,batch_size=batch_size,capacity=1000+3*batch_size,
		# 	min_after_dequeue=1000)
		#print(filename)
		image,label=read_and_decode(filename_queue)
		images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        capacity=1000 + 3 * batch_size,num_threads=1,
        min_after_dequeue=1000)
        #print("images shape:"+str(image))
        # print(image.shape)
        return images, sparse_labels

def train_input2crbm(log_name,conv_size,input_size,chanl_input,chanl_output):
	images_input,labels_input=inputs(train='train',batch_size=FLAGS.batch_size,
			num_epochs=FLAGS.num_epochs)
	print(images_input)
	#images=tf.reshape(images,[-1,input_size,input_size,chanl_input])
	W_conv1=tf.placeholder("float",[11,11,1,chanl_output])
	a_conv1=tf.placeholder("float",[1])
	b_conv1=tf.placeholder("float",[chanl_output])

	W_inc1=tf.placeholder("float",[11,11,1,chanl_output])
	a_inc1=tf.placeholder("float",[1])
	b_inc1=tf.placeholder("float",[chanl_output])

	images_placeholder=tf.placeholder(tf.float32,shape=(gd.BATCH_SIZE,gd.IMAGE_SIZE*gd.IMAGE_SIZE))

	images=tf.reshape(images_placeholder,[-1,gd.IMAGE_SIZE,gd.IMAGE_SIZE,1])

	pos_conv1_prob=1./(1+tf.exp(-conv2d_s4_valid(images,W_conv1)-b_conv1))
	pos_conv1_trans=tf.expand_dims(tf.reduce_mean(pos_conv1_prob,0),2)
	images_mean=tf.reduce_mean(images,0)
	images_trans=tf.expand_dims(tf.reshape(tf.transpose(tf.reshape(
	    tf.reduce_mean(images,0),[-1,1])),[1,gd.IMAGE_SIZE,gd.IMAGE_SIZE]),3)
	print('images_trans:'+str(images_trans))
	print('pos_conv1_trans:'+str(pos_conv1_trans))

	pos_prods_origin=conv2d_s16_valid(images_trans,pos_conv1_trans)
	print('pos_prods_origin:'+str(pos_prods_origin))

	pos_prods_trans=tf.transpose(pos_prods_origin,[1,2,0,3])

	pos_hid_act=tf.reduce_mean(pos_conv1_prob,0)

	pos_vis_act=tf.reduce_mean(images,0)
	#########################################################################3
	pos_hid_states=tf.to_float(tf.less_equal(tf.random_uniform(
	                shape=tf.shape(pos_conv1_prob)),pos_conv1_prob))


	# pos_hid_states_addpad=tf.pad(pos_hid_states,[[0,0],[conv_size-1,conv_size-1],[conv_size-1,conv_size-1],[0,0]],"CONSTANT")

	# W_transpose=tf.matrix_transpose(tf.reverse(W_conv1,[True,True,False,False]))


	# neg_data=1./(1+tf.exp(-conv2d_s1_valid(pos_hid_states_addpad,W_transpose)-a_conv1))

	neg_data=1./(1+tf.exp(-tf.nn.conv2d_transpose(pos_conv1_prob,W_conv1,[gd.BATCH_SIZE,gd.IMAGE_SIZE,gd.IMAGE_SIZE,1],strides=[1,4,4,1],padding='VALID')-a_conv1))
	print('neg_data:'+str(neg_data))

	neg_hid_probs=1./(1+tf.exp(-conv2d_s4_valid(neg_data,W_conv1)-b_conv1))

	neg_data_trans=tf.expand_dims(tf.reshape(tf.transpose(tf.reshape(
	    tf.reduce_mean(neg_data,0),[-1,1])),[1,gd.IMAGE_SIZE,gd.IMAGE_SIZE]),3)


	neg_hid_probs_trans=tf.expand_dims(tf.reduce_mean(neg_hid_probs,0),2)
	print("neg_data_trans:"+str(neg_data_trans))
	print("neg_hid_probs_trans:"+str(neg_hid_probs_trans))

	neg_prods_origin=conv2d_s16_valid(neg_data_trans,neg_hid_probs_trans)
	print('neg_prods_origin:'+str(neg_prods_origin))


	neg_prods_trans=tf.transpose(neg_prods_origin,[1,2,0,3])

	neg_hid_act=tf.reduce_mean(neg_hid_probs,0)
	neg_vis_act=tf.reduce_mean(neg_data,0)

	print('neg_hid_act:'+str(neg_hid_act))
	print('neg_vis_act:'+str(neg_vis_act))

	err_sum=tf.reduce_sum(tf.square(images-neg_data))

	reshaped_W=tf.transpose(tf.reshape(tf.transpose(tf.reshape(tf.squeeze(W_conv1),[-1,chanl_output])),[chanl_output,conv_size*conv_size]))



	W_inc_update=gd.momentum*W_inc1 + gd.epsilonw*((pos_prods_trans-neg_prods_trans)/gd.BATCH_SIZE - gd.weightcost*W_conv1)
	a_inc_update=gd.momentum*a_inc1 + (gd.epsilona/gd.BATCH_SIZE) * tf.reduce_mean(pos_vis_act-neg_vis_act)
	b_inc_update=gd.momentum*b_inc1 + (gd.epsilonb/gd.BATCH_SIZE) * tf.reduce_mean(tf.reduce_mean((pos_hid_act - neg_hid_act),0),0)
	

	init_op=tf.initialize_all_variables()
	tf.summary.scalar('loss',err_sum)
	tf.summary.scalar('a',a_conv1[0])
	tf.summary.scalar('b',b_conv1[0])
	tf.summary.scalar('W',W_conv1[0][0][0][0])


	summary_op=tf.summary.merge_all()


	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
		

	with tf.Session(config=config) as sess:
		  
		sess.run(init_op)
		# summary_writer=tf.train.SummaryWriter(FLAGS.train_dir,sess.graph)
		# coord=tf.train.Coordinator()
		# threads=tf.train.start_queue_runners(sess=sess,coord=coord)
		summary_writer=tf.summary.FileWriter(FLAGS.tensorevents_dir,sess.graph)
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(sess=sess,coord=coord)
		W_update_0=np.random.normal(0,0.01,[conv_size,conv_size,chanl_input,chanl_output])
		a_update_0=0.1*np.ones([chanl_input],np.float32)
		b_update_0=0.1*np.ones([chanl_output],np.float32)
		W_inc_update_0=np.zeros([conv_size,conv_size,chanl_input,chanl_output],np.float32)
		a_inc_update_0=np.zeros([chanl_input],np.float32)
		b_inc_update_0=np.zeros([chanl_output],np.float32)

		print("b_shape:"+str(b_update_0.shape))

		try:
			step=0
			while step<10000:
			# while not coord.should_stop():
				start_time=time.time()
				
				#print(images_input.eval(session=sess).shape)
				#print(a_update_0)
				
				# logfile=open(log_name,'a')
				# logfile.write("epoch: "+str(step)+'\n')
				# logfile.write("W:\n"+str(W_update_0[0])+'\n')

				images_wtf=images_input.eval(session=sess)


				# concat_img=Image.fromarray(
    #         tile_raster_images(
    #             X=images_wtf,
    #             img_shape=(32, 32),
    #             tile_shape=(10, 10)
    #         ))
				# concat_img.save(FLAGS.pic_dir+str(step)+'_train'+'.jpg')

				# logfile.close()

				loss,W_inc_update_0,a_inc_update_0,b_inc_update_0,neg_data_out,reshaped_W_out,\
				images_trans_out,pos_conv1_trans_out,neg_data_trans_out,neg_hid_probs_trans_out=sess.run(
					[err_sum,W_inc_update,a_inc_update,b_inc_update,neg_data,reshaped_W,
					images_trans,pos_conv1_trans,neg_data_trans,neg_hid_probs_trans],
					feed_dict={images_placeholder:images_wtf,
					W_conv1:W_update_0,a_conv1:a_update_0,b_conv1:b_update_0,W_inc1:W_inc_update_0,
					a_inc1:a_inc_update_0,b_inc1:b_inc_update_0})

				W_update_0=W_update_0+W_inc_update_0
				a_update_0=a_update_0+a_inc_update_0
				b_update_0=b_update_0+b_inc_update_0

				#print("step %d: loss = %f" %(step,loss))
				# logfile=open(log_name,'a')
				# logfile.write('step '+str(step)+": loss="+str(loss)+'\n')
				# # logfile.write("W_inc:\n"+str(W_inc_update_0[0])+'\n')
				# # logfile.write("a_inc:\n"+str(a_inc_update_0[0])+'\n')
				# # logfile.write("b_inc:\n"+str(b_inc_update_0[0])+'\n')
				# # logfile.write("neg_data:\n"+str(neg_data_out[0])+'\n')
				# # logfile.write("reshaped_W:\n"+str(reshaped_W_out[0])+'\n')
				# # logfile.write("pos_conv1_prob_out:\n"+str(pos_conv1_prob_out[0])+'\n')
				# logfile.write("images_trans:\n"+str(images_trans_out[0][0])+'\n')
				# logfile.write("pos_conv1_trans:\n"+str(pos_conv1_trans_out[0][0])+'\n')
				# #logfile.write("neg_hid_probs_out:\n"+str(neg_hid_probs_out[0])+'\n')
				# logfile.write("neg_data_trans_out:\n"+str(neg_data_trans_out[0][0])+'\n')
				# logfile.write("neg_hid_probs_trans:\n"+str(neg_hid_probs_trans_out[0][0])+'\n')
				# logfile.write("W_inc:\n"+str(W_update_0[0])+'\n')
				# logfile.close()

				# logfile=open(log_name,'a')
				# logfile.write("epoch: "+str(step)+'\n')
				# logfile.write("W_inc:\n"+str(W_inc_update_0[0])+'\n')

				# logfile.close()

				#print('step '+str(step)+": loss="+str(loss)+'\n')

				if step % 10== 0:
					print("step %d: loss = %f" %(step,loss))
					logfile=open(log_name,'a')
					logfile.write('step '+str(step)+": loss="+str(loss)+'\n')
					logfile.close()
					#print(neg_data_out.shape)
					
					# weight_img=Image.fromarray(
     #        tile_raster_images(
     #            X=reshaped_W_out.T,
     #            img_shape=(conv_size, conv_size),
     #            tile_shape=(8, 12),
     #        ))
					# weight_img.save(FLAGS.Weight_dir+'weight_'+str(step)+'.jpg')

					summary_str = sess.run(summary_op,
						feed_dict={images_placeholder:images_input.eval(session=sess),
					W_conv1:W_update_0,a_conv1:a_update_0,b_conv1:b_update_0,W_inc1:W_inc_update_0,
					a_inc1:a_inc_update_0,b_inc1:b_inc_update_0})
					summary_writer.add_summary(summary_str, step)
					
					if step % 50==0:
						saveimg=Image.fromarray(255*to_image(images_wtf.reshape(gd.BATCH_SIZE,input_size,input_size,1))[:,:,0])
						#print(to_image(images_input.eval(session=sess).reshape(gd.BATCH_SIZE,input_size,input_size,1)).shape)
						saveimg=saveimg.convert('RGB')
						saveimg.save(FLAGS.pic_dir+'imag_layer1_epoch'+str(step)+'.jpg')

						saveimg_negv=Image.fromarray(255*to_image(neg_data_out)[:,:,0])
						#print(to_image(neg_data_out).shape)
						saveimg_negv=saveimg_negv.convert('RGB')
						saveimg_negv.save(FLAGS.pic_dir+'negv_layer1_epoch'+str(step)+'.jpg')

						print('Here 2')
						save_fn=FLAGS.log_dir+'/parameters_layer1_epoch_'+str(step)+'.mat'
						sio.savemat(save_fn,{'W1':W_update_0,'b1':b_update_0})

				step+=1
				# if step==10:
				# 	return W_update_0,a_update_0,b_update_0
		except tf.errors.OutOfRangeError:
			print('Done training for %d epochs, %d steps.' % (1001, step))
		finally:
			coord.request_stop()
			
		coord.request_stop()
		coord.join(threads)
		sess.close()
		return W_update_0,a_update_0,b_update_0

def train_crbm2crbm(log_name,conv_size,input_size,chanl_input,chanl_output,parameters):
	images_input,labels_input=inputs(train='train',batch_size=FLAGS.batch_size,
			num_epochs=FLAGS.num_epochs)
	#images=tf.reshape(images,[-1,input_size,input_size,chanl_input])
	W_conv1=tf.placeholder("float",[conv_size,conv_size,chanl_input,chanl_output])
	a_conv1=tf.placeholder("float",[chanl_input])
	b_conv1=tf.placeholder("float",[chanl_output])

	W_inc1=tf.placeholder("float",[conv_size,conv_size,chanl_input,chanl_output])
	a_inc1=tf.placeholder("float",[chanl_input])
	b_inc1=tf.placeholder("float",[chanl_output])

	W_extra1=tf.placeholder("float",[11,11,1,64])
	#a_extra1=tf.placeholder("float",[1])
	b_extra1=tf.placeholder("float",[64])
	###############LAYER2############################
	W_extra2=tf.placeholder("float",[5,5,64,192])
	b_extra2=tf.placeholder("float",[192])
	# #################LAYER3#########################################
	W_extra3=tf.placeholder("float",[3,3,192,384])
	b_extra3=tf.placeholder("float",[384])
	# #################LAYER4################################################
	W_extra4=tf.placeholder("float",[3,3,384,384])
	b_extra4=tf.placeholder("float",[384])

	images_placeholder=tf.placeholder(tf.float32,shape=(gd.BATCH_SIZE,gd.IMAGE_SIZE*gd.IMAGE_SIZE))
	images_extra=tf.reshape(images_placeholder,[-1,gd.IMAGE_SIZE,gd.IMAGE_SIZE,1])

	h_conv1=1./(1+tf.exp(-conv2d(images_extra,W_extra1,4,'VALID')-b_extra1))
	#norm1=tf.nn.lrn(h_conv1,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm1')
	h_pool1=max_pool(h_conv1,3,2,'VALID')
	#images=h_pool1
###################LAYER2#######################################################
	h_conv2=1./(1+tf.exp(-conv2d(h_pool1,W_extra2,1,'SAME')-b_extra2))
	#norm2=tf.nn.lrn(h_conv2,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm2')
	h_pool2=max_pool(h_conv2,3,2,'VALID')
# 	#images=h_pool2
######################LAYER3####################################################
	h_conv3=1./(1+tf.exp(-conv2d(h_pool2,W_extra3,1,'SAME')-b_extra3))
	#images=h_conv3
# ######################LAYER4#######################################################
	h_conv4=1./(1+tf.exp(-conv2d(h_conv3,W_extra4,1,'SAME')-b_extra4))


	images=h_conv4
	print(images)
	

	pos_conv1_prob=1./(1+tf.exp(-conv2d_s1_valid(images,W_conv1)-b_conv1))
	pos_conv1_trans=tf.expand_dims(tf.reduce_mean(pos_conv1_prob,0),2)
	images_mean=tf.reduce_mean(images,0)
	images_trans=tf.expand_dims(tf.reshape(tf.transpose(tf.reshape(
		tf.reduce_mean(images,0),[-1,chanl_input])),[chanl_input,input_size,input_size]),3)

	pos_prods_origin=conv2d_s1_valid(images_trans,pos_conv1_trans)

	pos_prods_trans=tf.transpose(pos_prods_origin,[1,2,0,3])

	pos_hid_act=tf.reduce_mean(pos_conv1_prob,0)

	pos_vis_act=tf.reduce_mean(images,0)
	#########################################################################3
	pos_hid_states=tf.to_float(tf.less_equal(tf.random_uniform(
					shape=tf.shape(pos_conv1_prob)),pos_conv1_prob))


	pos_hid_states_addpad=tf.pad(pos_hid_states,[[0,0],[conv_size-1,conv_size-1],[conv_size-1,conv_size-1],[0,0]],"CONSTANT")

	#W_transpose=tf.matrix_transpose(tf.reverse(W_conv1,[True,True,False,False]))
	W_transpose=tf.matrix_transpose(tf.reverse(W_conv1,[0,1]))


	neg_data=1./(1+tf.exp(-conv2d_s1_valid(pos_hid_states_addpad,W_transpose)-a_conv1))

	neg_hid_probs=1./(1+tf.exp(-conv2d_s1_valid(neg_data,W_conv1)-b_conv1))

	neg_data_trans=tf.expand_dims(tf.reshape(tf.transpose(tf.reshape(
		tf.reduce_mean(neg_data,0),[-1,chanl_input])),[chanl_input,input_size,input_size]),3)


	neg_hid_probs_trans=tf.expand_dims(tf.reduce_mean(neg_hid_probs,0),2)

	neg_prods_origin=conv2d_s1_valid(neg_data_trans,neg_hid_probs_trans)

	neg_prods_trans=tf.transpose(neg_prods_origin,[1,2,0,3])

	neg_hid_act=tf.reduce_mean(neg_hid_probs,0)
	neg_vis_act=tf.reduce_mean(neg_data,0)

	err_sum=tf.reduce_sum(tf.square(images-neg_data))

	#reshaped_W=tf.transpose(tf.reshape(tf.transpose(tf.reshape(tf.squeeze(W_conv1),[-1,chanl_output])),[chanl_output*chanl_input,conv_size*conv_size]))



	W_inc_update=gd.momentum*W_inc1 + gd.epsilonw*((pos_prods_trans-neg_prods_trans)/gd.BATCH_SIZE - gd.weightcost*W_conv1)
	a_inc_update=gd.momentum*a_inc1 + (gd.epsilona/gd.BATCH_SIZE) * tf.reduce_mean(pos_vis_act-neg_vis_act)
	b_inc_update=gd.momentum*b_inc1 + (gd.epsilonb/gd.BATCH_SIZE) * tf.reduce_mean(tf.reduce_mean((pos_hid_act - neg_hid_act),0),0)
	

	init_op=tf.initialize_all_variables()
	tf.summary.scalar('loss',err_sum)
	tf.summary.scalar('a',a_conv1[0])
	tf.summary.scalar('b',b_conv1[0])
	tf.summary.scalar('W',W_conv1[0][0][0][0])


	summary_op=tf.summary.merge_all()


	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
		

	with tf.Session(config=config) as sess:
		  
		sess.run(init_op)
		# summary_writer=tf.train.SummaryWriter(FLAGS.train_dir,sess.graph)
		# coord=tf.train.Coordinator()
		# threads=tf.train.start_queue_runners(sess=sess,coord=coord)
		summary_writer=tf.summary.FileWriter(FLAGS.tensorevents_dir,sess.graph)
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(sess=sess,coord=coord)
		W_update_0=np.random.normal(0,0.01,[conv_size,conv_size,chanl_input,chanl_output])
		a_update_0=0.1*np.ones([chanl_input],np.float32)
		b_update_0=0.1*np.ones([chanl_output],np.float32)
		W_inc_update_0=np.zeros([conv_size,conv_size,chanl_input,chanl_output],np.float32)
		a_inc_update_0=np.zeros([chanl_input],np.float32)
		b_inc_update_0=np.zeros([chanl_output],np.float32)

		W_extra1_0=parameters[0]
		print(W_extra1_0.shape)
		b_extra1_0=parameters[1].reshape(parameters[1].shape[1])
		#b_extra1_0=np.transpose(parameters[1])
		print(b_extra1_0.shape)
		################LAYER2##########################################
		W_extra2_0=parameters[2]
		#b_extra2_0=np.transpose(parameters[3])
		b_extra2_0=parameters[3].reshape(parameters[3].shape[1])
		# # #print(b_extra2_0.shape)
		W_extra3_0=parameters[4]
		b_extra3_0=parameters[5].reshape(parameters[5].shape[1])

		W_extra4_0=parameters[6]
		b_extra4_0=parameters[7].reshape(parameters[7].shape[1])

		try:
			step=0

			while step<10000:
				start_time=time.time()
				
				#print(images_input.eval(session=sess).shape)
				#print(a_update_0)
				
				# logfile=open(log_name,'a')
				# logfile.write("epoch: "+str(step)+'\n')
				# logfile.write("W:\n"+str(W_update_0[0])+'\n')

				images_wtf=images_input.eval(session=sess)


				# concat_img=Image.fromarray(
    #         tile_raster_images(
    #             X=images_wtf,
    #             img_shape=(32, 32),
    #             tile_shape=(10, 10)
    #         ))
				# concat_img.save(FLAGS.pic_dir+str(step)+'_train'+'.jpg')

				# logfile.close()

				W_inc_update_0,a_update_0,b_inc_update_0,loss,neg_data_out,images_out=sess.run(
					[W_inc_update,a_inc_update,b_inc_update,err_sum,neg_data,images],
					feed_dict={images_placeholder:images_wtf,
					W_conv1:W_update_0,a_conv1:a_update_0,b_conv1:b_update_0,W_inc1:W_inc_update_0,
					a_inc1:a_inc_update_0,b_inc1:b_inc_update_0,
					W_extra1:W_extra1_0,b_extra1:b_extra1_0,
					W_extra2:W_extra2_0,b_extra2:b_extra2_0,
					W_extra3:W_extra3_0,b_extra3:b_extra3_0,
					W_extra4:W_extra4_0,b_extra4:b_extra4_0})

				W_update_0=W_update_0+W_inc_update_0
				a_update_0=a_update_0+a_inc_update_0
				b_update_0=b_update_0+b_inc_update_0

				# logfile=open(log_name,'a')
				# logfile.write("epoch: "+str(step)+'\n')
				# logfile.write("W_inc:\n"+str(W_inc_update_0[0])+'\n')

				# logfile.close()

				#print('step '+str(step)+": loss="+str(loss)+'\n')
				#print("step %d: loss = %d" %(step,loss))




				if step % 10== 0:
					print("step %d: loss = %d" %(step,loss))
					logfile=open(log_name,'a')
					logfile.write('step '+str(step)+": loss="+str(loss)+'\n')
					#logfile.write("W:\n"+str(W_update_0[0])+'\n')
					#logfile.write("W_inc:\n"+str(W_inc_update_0[0])+'\n')
					logfile.close()

					
					#print(neg_data_out.shape)
					#print("images_wtf shape"+str(images_wtf.shape))
					# saveimg=Image.fromarray(255*to_image(images_wtf.reshape(gd.BATCH_SIZE,input_size,input_size,chanl_input))[:,:,0])
					# #print(to_image(images_input.eval(session=sess).reshape(gd.BATCH_SIZE,input_size,input_size,1)).shape)
					# saveimg=saveimg.convert('RGB')
					# saveimg.save(FLAGS.pic_dir+'imag_layer1_epoch'+str(step)+'.jpg')

					# saveimg_negv=Image.fromarray(255*to_image(neg_data_out)[:,:,0])
					# #print(to_image(neg_data_out).shape)
					# saveimg_negv=saveimg_negv.convert('RGB')
					# saveimg_negv.save(FLAGS.pic_dir+'negv_layer1_epoch'+str(step)+'.jpg')

					# weight_img=Image.fromarray(
     #        tile_raster_images(
     #            X=reshaped_W_out.T,
     #            img_shape=(conv_size, conv_size),
     #            tile_shape=(chanl_input, chanl_output),
     #        ))
					# weight_img.save(FLAGS.Weight_dir+'weight_'+str(step)+'.jpg')

					summary_str = sess.run(summary_op,
						feed_dict={images_placeholder:images_wtf,
					W_conv1:W_update_0,a_conv1:a_update_0,b_conv1:b_update_0,W_inc1:W_inc_update_0,
					a_inc1:a_inc_update_0,b_inc1:b_inc_update_0,
					W_extra1:W_extra1_0,b_extra1:b_extra1_0,
					W_extra2:W_extra2_0,b_extra2:b_extra2_0,
					W_extra3:W_extra3_0,b_extra3:b_extra3_0,
					W_extra4:W_extra4_0,b_extra4:b_extra4_0})
					summary_writer.add_summary(summary_str, step)
						
					if step % 50==0:
						save_fn=FLAGS.log_dir+'/parameters_layer4_epoch_'+str(step)+'.mat'
						sio.savemat(save_fn,{'W1':W_extra1_0,'b1':b_extra1_0,
						'W2':W_extra2_0,'b2':b_extra2_0,
						'W3':W_extra3_0,'b3':b_extra3_0,
						'W4':W_extra4_0,'b4':b_extra4_0,
						'W5':W_update_0,'b5':b_update_0})
						saveimg=Image.fromarray(255*to_image(images_out)[:,:,0])
					#print(to_image(images_input.eval(session=sess).reshape(gd.BATCH_SIZE,input_size,input_size,in)).shape)
						saveimg=saveimg.convert('RGB')
						saveimg.save(FLAGS.pic_dir+'imag_layer1_epoch'+str(step)+'.jpg')

						saveimg_negv=Image.fromarray(255*to_image(neg_data_out)[:,:,0])
						#print(to_image(neg_data_out).shape)
						saveimg_negv=saveimg_negv.convert('RGB')
						saveimg_negv.save(FLAGS.pic_dir+'negv_layer1_epoch'+str(step)+'.jpg')

				step+=1
				# if step==100:
				# 	return W_update_0,a_update_0,b_update_0
		except tf.errors.OutOfRangeError:
			print('Done training for %d epochs, %d steps.' % (1001, step))
		finally:
			coord.request_stop()

		coord.join(threads)
		sess.close()
		return W_update_0,a_update_0,b_update_0

if __name__=="__main__":
	# log_name=str(FLAGS.log_dir)+'zooscan_224_224_20_'+'layer1'+'_'+re.sub(r'[^0-7]','',str(datetime.datetime.now()))+'.txt'
	# f=open(log_name,'w')
	# f.close()
	# # print('here 1')
	# # train_input2crbm(log_name,conv_size,input_size,chanl_input,chanl_output)
	# W1,a1,b1=train_input2crbm(log_name,11,224,1,64)
	# print("returned b1 shape:"+str(b1.shape))
	# save_fn=FLAGS.log_dir+'/parameters_layer1.mat'
	# sio.savemat(save_fn,{'W1':W1,'a1':a1,'b1':b1})
	# # #print(b1)

	####################Conv-Layer2################################################################

	# log_name=str(FLAGS.log_dir)+'zooscan_224_224_20_'+'layer2'+'_'+str(time_value)+'.txt'
	# load_fn='/home/scw4750/Liuhongkun/tfrecord/zooscan/Alex_cdbn/data_record/experiment4/log201702217203021215/parameters_layer1_epoch_350.mat'
	# load_data=sio.loadmat(load_fn)
	# W1=load_data['W1']
	# #a1=load_data['a1']
	# b1=load_data['b1']
	# parameters_layer1=[W1,b1]
	# #print("reloaded b1 shape:"+str(b1.shape))
	# conv_size=5
	# input_size=26
	# chanl_input=64
	# chanl_output=192
	# W2,a2,b2=train_crbm2crbm(log_name,conv_size,input_size,chanl_input,chanl_output,parameters_layer1)
	# # # # print(a2)
	# save_fn='/home/scw4750/Liuhongkun/tfrecord/kaggle_zooplankton/cdbn2/parameter_saved/vs_lenet/experiment1/parameters_layer2.mat'
	# sio.savemat(save_fn,{'W1':W1,'a1':a1,'b1':b1,
	# 					'W2':W2,'a2':a2,'W3':b2})
	###################Conv-Layer3################################################################
# 	log_name=str(FLAGS.log_dir)+'zooscan_224_224_20'+'layer3'+'.txt'
# 	load_fn='/home/scw4750/Liuhongkun/tfrecord/zooscan/Alex_cdbn/data_record/experiment4/LAYER2/log201702210504330047/parameters_layer4_epoch_450.mat'
# 	load_data=sio.loadmat(load_fn)

# 	W1=load_data['W1']
# 	#a1=load_data['a1']
# 	b1=load_data['b1']
# 	W2=load_data['W2']
# 	b2=load_data['b2']
# 	parameters_layer=[W1,b1,W2,b2]
# 	#print("reloaded b1 shape:"+str(b1.shape))
# 	conv_size=3
# 	input_size=12
# 	chanl_input=192
# 	chanl_output=384
# 	W2,a2,b2=train_crbm2crbm(log_name,conv_size,input_size,chanl_input,chanl_output,parameters_layer)
# ####################LAYER4############################################################################	
# 	log_name=str(FLAGS.log_dir)+'zooscan_224_224_20_'+'layer4'+'.txt'
# 	load_fn='/home/scw4750/Liuhongkun/tfrecord/zooscan/Alex_cdbn/data_record/experiment4/LAYER3/log2017022122237724/parameters_layer4_epoch_1900.mat'
# 	load_data=sio.loadmat(load_fn)

# 	W1=load_data['W1']
# 	#a1=load_data['a1']
# 	b1=load_data['b1']
# 	W2=load_data['W2']
# 	b2=load_data['b2']
# 	W3=load_data['W3']
# 	b3=load_data['b3']
# 	parameters_layer=[W1,b1,W2,b2,W3,b3]
# 	#print("reloaded b1 shape:"+str(b1.shape))
# 	conv_size=3
# 	input_size=12
# 	chanl_input=384
# 	chanl_output=384
# 	W2,a2,b2=train_crbm2crbm(log_name,conv_size,input_size,chanl_input,chanl_output,parameters_layer)

# ###################LAYER5#################################################
	log_name=str(FLAGS.log_dir)+'zooscan_224_224_20_'+'layer5'+'.txt'
	load_fn='/home/scw4750/Liuhongkun/tfrecord/zooscan/Alex_cdbn/data_record/experiment2/Layer4/log201702271334110063/parameters_layer4_epoch_700.mat'
	load_data=sio.loadmat(load_fn)

	W1=load_data['W1']
	#a1=load_data['a1']
	b1=load_data['b1']
	W2=load_data['W2']
	b2=load_data['b2']
	W3=load_data['W3']
	b3=load_data['b3']
	W4=load_data['W4']
	b4=load_data['b4']
	parameters_layer=[W1,b1,W2,b2,W3,b3,W4,b4]
	#print("reloaded b1 shape:"+str(b1.shape))
	conv_size=3
	input_size=12
	chanl_input=384
	chanl_output=256
	W2,a2,b2=train_crbm2crbm(log_name,conv_size,input_size,chanl_input,chanl_output,parameters_layer)
