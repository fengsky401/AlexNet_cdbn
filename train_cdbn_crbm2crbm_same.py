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
	'/home/scw4750/Liuhongkun/tfrecord/zooscan/tfrecord/227_227_20_modified/', 'Directory to put the training data.')
flags.DEFINE_string('filename', 'train.tfrecords', 'Directory to put the training data.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
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



	err_sum=tf.reduce_sum(tf.square(images-neg_data))

	#reshaped_W=tf.transpose(tf.reshape(tf.transpose(tf.reshape(tf.squeeze(W_conv1),[-1,chanl_output])),[chanl_output*chanl_input,conv_size*conv_size]))



	W_inc_update=gd.momentum*W_inc1 + gd.epsilonw*((pos_prods_trans-neg_prods_trans)/gd.BATCH_SIZE - weightcost*W_conv1)
	a_inc_update=gd.momentum*a_inc1 + (gd.epsilona/gd.BATCH_SIZE) * tf.reduce_mean(pos_vis_act-neg_vis_act)
	b_inc_update=gd.momentum*b_inc1 + (gd.epsilonb/gd.BATCH_SIZE) * tf.reduce_mean(tf.reduce_mean((pos_hid_act - neg_hid_act),0),0)
	

	init_op=tf.initialize_all_variables()
	tf.scalar_summary('loss',err_sum)
	tf.scalar_summary('a',a_conv1[0])
	tf.scalar_summary('b',b_conv1[0])
	tf.scalar_summary('W',W_conv1[0][0][0][0])


	summary_op=tf.merge_all_summaries()


	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
		

	with tf.Session(config=config) as sess:
		  
		sess.run(init_op)
		# summary_writer=tf.train.SummaryWriter(FLAGS.train_dir,sess.graph)
		# coord=tf.train.Coordinator()
		# threads=tf.train.start_queue_runners(sess=sess,coord=coord)
		summary_writer=tf.train.SummaryWriter(FLAGS.tensorevents_dir,sess.graph)
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(sess=sess,coord=coord)
		W_update_0=np.random.normal(0,0.1,[conv_size,conv_size,chanl_input,chanl_output])
		a_update_0=np.zeros([chanl_input],np.float32)
		b_update_0=np.zeros([chanl_output],np.float32)
		W_inc_update_0=np.zeros([conv_size,conv_size,chanl_input,chanl_output],np.float32)
		a_inc_update_0=np.zeros([chanl_input],np.float32)
		b_inc_update_0=np.zeros([chanl_output],np.float32)

		W_extra1_0=parameters[0]
		a_extra1_0=parameters[1]
		b_extra1_0=parameters[2].reshape(chanl_input)

		try:
			step=0

			while step<3000:
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
					W_extra1:W_extra1_0,b_extra1:b_extra1_0})

				W_update_0=W_update_0+W_inc_update_0
				a_update_0=a_update_0+a_inc_update_0
				b_update_0=b_update_0+b_inc_update_0

				# logfile=open(log_name,'a')
				# logfile.write("epoch: "+str(step)+'\n')
				# logfile.write("W_inc:\n"+str(W_inc_update_0[0])+'\n')

				# logfile.close()

				#print('step '+str(step)+": loss="+str(loss)+'\n')
				print("step %d: loss = %d" %(step,loss))



				if step % 10== 0:
					logfile=open(log_name,'a')
					logfile.write('step '+str(step)+": loss="+str(loss)+'\n')
					logfile.write("W:\n"+str(W_update_0[0])+'\n')
					#logfile.write("W_inc:\n"+str(W_inc_update_0[0])+'\n')
					logfile.close()

					
					# print(to_image(neg_data_out).shape)
					# print("images_out"+str(to_image(images_out).shape))
				

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
					W_extra1:W_extra1_0,b_extra1:b_extra1_0})
					summary_writer.add_summary(summary_str, step)
						
					if step % 50==0:
						save_fn=FLAGS.log_dir+'/parameters_layer2_epoch_'+str(step)+'.mat'
						sio.savemat(save_fn,{'W1':W_extra1_0,'b1':b_extra1_0,
						'W2':W_update_0,'b2':b_update_0})
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
	# log_name=str(FLAGS.log_dir)+'zooscan_227_227_20modified_'+'layer1'+'_'+re.sub(r'[^0-7]','',str(datetime.datetime.now()))+'.txt'
	# f=open(log_name,'w')
	# f.close()
	# # print('here 1')
	# # train_input2crbm(log_name,conv_size,input_size,chanl_input,chanl_output)
	# W1,a1,b1=train_input2crbm(log_name,11,227,1,96)
	# print("returned b1 shape:"+str(b1.shape))
	# save_fn=FLAGS.log_dir+'/parameters_layer1.mat'
	# sio.savemat(save_fn,{'W1':W1,'a1':a1,'b1':b1})
	# # #print(b1)

	log_name=str(FLAGS.log_dir)+'zooscan_227_227_20modified_'+'layer2'+'.txt'
	load_fn='/home/scw4750/Liuhongkun/tfrecord/zooscan/Alex_cdbn/data_record/experiment1/layer1/log2017021173725417716/parameters_layer1_epoch_200.mat'
	load_data=sio.loadmat(load_fn)
	W1=load_data['W1']
	a1=load_data['a1']
	b1=load_data['b1']
	parameters_layer1=[W1,a1,b1]
	#print("reloaded b1 shape:"+str(b1.shape))
	#train_crbm2crbm(log_name,conv_size,input_size,chanl_input,chanl_output,parameters)
	conv_size=5
	input_size=27
	chanl_input=96
	chanl_output=256
	W2,a2,b2=train_crbm2crbm(log_name,conv_size,input_size,chanl_input,chanl_output,parameters_layer1)
	# # # print(a2)
	# # save_fn='/home/scw4750/Liuhongkun/tfrecord/kaggle_zooplankton/cdbn2/parameter_saved/vs_lenet/experiment1/parameters_layer2.mat'
	# # sio.savemat(save_fn,{'W1':W1,'a1':a1,'b1':b1,
	# # 					'W2':W2,'a2':a2,'W3':b2})