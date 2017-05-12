#######################################################################################################
#将train_cdbn.py生成的卷积波尔兹曼机的自训练结果带入，初始化前5个卷积层，然后用后向传播的方法对参数进行微调。##################3
#########################################################################################################3333

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np
import datetime
import os
import re

import tensorflow as tf

import global_define as gd
import Alexnet
import PIL
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from PIL import Image
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
#from sklearn import datasets, metrics, cross_validation
import sklearn as sk
#from tensorflow.models.inception.inception.slim import scopes 

from utils import tile_raster_images
from tensorflow.contrib.slim.nets import alexnet
slim = tf.contrib.slim
alexnet = tf.contrib.slim.nets.alexnet
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE='test.tfrecords'

flags = tf.app.flags
FLAGS = flags.FLAGS
#FLAGS = None
time_value=re.sub(r'[^0-7]','',str(datetime.datetime.now()))
flags.DEFINE_string('tfrecord_dir', 
	'/media/scw4750/25a01ed5-a903-4298-87f2-a5836dcb6888/Liuhongkun/zooscan_2/shuffled/224_224/', 'Directory to put the training data.')
flags.DEFINE_string('filename', 'train.tfrecords', 'Directory to put the training data.')
flags.DEFINE_integer('batch_size',100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('num_epochs', None, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')

flags.DEFINE_integer('learning_rate', 0.02,'balabala')
flags.DEFINE_integer('max_steps', 50000,'balabala')
flags.DEFINE_string('model_dir','Modal/model'+str(time_value)+'/','balabala')
flags.DEFINE_string('tensorevents_dir','tensorboard_event/event_wth'+str(time_value)+'/','balabala')
flags.DEFINE_string('log_dir','Log_data/log'+str(time_value)+'/','balabala')
flags.DEFINE_string('pic_dir','Pic/Pictures_input'+str(time_value)+'/','balabala')


if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

if not os.path.exists(FLAGS.tensorevents_dir):
  os.makedirs(FLAGS.tensorevents_dir)

if not os.path.exists(FLAGS.model_dir):
	os.makedirs(FLAGS.model_dir)

if not os.path.exists(FLAGS.pic_dir):
	os.makedirs(FLAGS.pic_dir)


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

	precision=float(true_count)/FLAGS.batch_size/FLAGS.batch_size
	# print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
 #            (FLAGS.batch_size, true_count, precision))
 	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f\n' %
            (FLAGS.batch_size*FLAGS.batch_size, true_count, precision))
	logfile=open(log_name,'a')
	logfile.write('  Num examples: %d  Num correct: %d  Precision : %0.04f\n' %
            (FLAGS.batch_size, true_count, precision))
	
	logfile.close()
	return precision

def inputs(train,batch_size,num_epochs):
	if not num_epochs:num_epochs=None
	if train=='train':
		filename=os.path.join(FLAGS.tfrecord_dir,gd.TRAIN_FILE)
	elif train=='validation':
		filename=os.path.join(FLAGS.tfrecord_dir,gd.VALIDATION_FILE)
	else:
		filename=os.path.join(FLAGS.tfrecord_dir,gd.TEST_FILE)

	with tf.name_scope('input'):
		filename_queue=tf.train.string_input_producer([filename],num_epochs=None)
		print(filename)
		image,label=read_and_decode(filename_queue)
		print(image)
		images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=1,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)
        return images, sparse_labels

def run_training(log_name):
	with tf.Graph().as_default():
	   	#with slim.arg_scope(alexnet.alexnet_v2_arg_scope()) as scope:
	   	with tf.variable_scope("Alexnet") as scope:
	   	#with slim.arg_scope() as scope:

			images,labels=inputs(train='train',batch_size=FLAGS.batch_size,
			num_epochs=FLAGS.num_epochs)

		
			images_test,labels_test=inputs(train='test', batch_size=FLAGS.batch_size,
                             num_epochs=FLAGS.num_epochs)

			images=tf.reshape(images,[-1,gd.INPUT_SIZE,gd.INPUT_SIZE,1])
			# batch_size=tf.size(labels)
			# labels=tf.expand_dims(labels,1)
			# indices=tf.expand_dims(tf.range(0,batch_size),1)
			# print('indices:'+str(indices))
			# print('labels:'+str(labels))
			# #concated=tf.concat(1,[indices,labels])
			# concated=tf.concat([indices,labels_test],1)
			# onehot_labels_test=tf.sparse_to_dense(concated,tf.stack([batch_size,gd.NUM_CLASSES]),1.0,0.0)

			# validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
   #                    "precision": tf.contrib.metrics.streaming_precision,
   #                    "recall": tf.contrib.metrics.streaming_recall}
			# validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
			#     images_test,
			#     labels_test,
			#     every_n_steps=100,
			#     metrics=validation_metrics)

			logits,description = Alexnet.inference(images)

			# yp_train = tf.argmax(logits, 1)
			# print("yp_train:"+str(yp_train))
			# print("labels:"+str(labels))
			# #print(description)
		
			#logits=Alexnet.inference(images)
			tf.get_variable_scope().reuse_variables()

			images_test=tf.reshape(images_test,[-1,gd.INPUT_SIZE,gd.INPUT_SIZE,1])
			
			logits_test,_=Alexnet.inference(images_test)
			#yp_test = tf.argmax(logits_test, 1)
			# print("yp_train:"+str(yp_train))
			# print("labels:"+str(labels))

			loss=Alexnet.loss(logits,labels)

			train_op=Alexnet.training(loss,FLAGS.learning_rate)

			eval_correct=Alexnet.evaluation(logits,labels)

			eval_correct_test=Alexnet.evaluation(logits_test,labels_test)

			confusion_mat = Alexnet.confusion_matrix(logits_test,labels_test)
			
			# recall_value=Alexnet.recall(logits,labels)
			
			# recall_value_test=Alexnet.recall(logits_test,labels_test)

			# accuracy_value=Alexnet.accuracy(logits,labels)

			# accuracy_test=Alexnet.accuracy(logits_test,labels_test)

			summary_op=tf.summary.merge_all()

		init_op=tf.initialize_all_variables()
		
		saver=tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True

		with tf.Session(config=config) as sess:

			sess.run(init_op)
			summary_writer=tf.summary.FileWriter(FLAGS.tensorevents_dir,sess.graph)
			coord=tf.train.Coordinator()
			threads=tf.train.start_queue_runners(sess=sess,coord=coord)
			try:
				step=0
				# checkpoint_file = os.path.join(FLAGS.model_dir, 'model.ckpt')
				# saver.save(sess, checkpoint_file, global_step=step)

				while not coord.should_stop():
					start_time=time.time()
					_,loss_value,images_out=sess.run([train_op,loss,images])

					duration=time.time()-start_time
					if step % 100 == 0:

						#cofmat=sess.run(confusion_mat)
						summary_str=sess.run(summary_op)
						summary_writer.add_summary(summary_str,step)
						
						# concat_img=Image.fromarray(
      #       tile_raster_images(
      #           X=images_out,
      #           img_shape=(227, 227),
      #           tile_shape=(10, 10)
      #       ))
						# concat_img.save(FLAGS.pic_dir+str(step)+'_train'+'.jpg')

						print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
						                                 duration))
						logfile=open(log_name,'a')
						logfile.write('Step %d: loss = %.2f (%.3f sec)\n' % (step, loss_value, duration))
						logfile.close()
					if (step ) % 1000 == 0 or (step ) == FLAGS.max_steps:
						# checkpoint_file = str(FLAGS.model_dir)+'/'+'mymodel'
						# saver.save(sess, checkpoint_file)

						#saver.restore(sess,checkpoint_file)
						
						#print("recall:%f" %(cofmat))
						# ypred_test,labels_test=sess.run([yp_test,labels_test])
						# print("Train: sk.precision: %f" %(sk.metrics.precision_score(ypred_test,labels_test)))
						# print("Train: sk.recall: %f" %(sk.metrics.recall_score(ypred_test,labels_test)))
						# print("Train: sk.f1_score: %f" %(sk.metrics.f1_score(ypred_test,labels_test)))
						# print("Train: confusion_matrix:")
						# print(sk.metrics.confusion_matrix(ypred_test,labels_test))
						# false_positive_rate, true_positive_rate, thresholds = roc_curve(ypred_train, ypred_train)
						# roc_auc = auc(false_positive_rate, true_positive_rate)
						# plt.title('Receiver Operating Characteristic')
						# plt.plot(false_positive_rate, true_positive_rate, 'b',
						# label='AUC = %0.2f'% roc_auc)
						# plt.legend(loc='lower right')
						# plt.plot([0,1],[0,1],'r--')
						# plt.xlim([-0.1,1.2])
						# plt.ylim([-0.1,1.2])
						# plt.ylabel('True Positive Rate')
						# plt.xlabel('False Positive Rate')
						# plt.show()
						print('Train:')
						logfile=open(log_name,'a')

						logfile.write('Train:\n')
						logfile.close()
						do_eval(sess,eval_correct,log_name)
						# print('Validation:')
						# do_eval(sess,eval_correct_valid,log_name)
						print('Test:')
						logfile=open(log_name,'a')
						logfile.write('Test:\n')
						logfile.close()
						precision_test=do_eval(sess,eval_correct_test,log_name)
						#tf.scalar_summary("accuracy_test",precision_test)
						summary_str=sess.run(summary_op)
						summary_writer.add_summary(summary_str,step)
					if step % 10000==0:
						print("Saving Model done")
						checkpoint_file = str(FLAGS.model_dir)+'/'+'mymodel'+str(step)
						saver.save(sess, checkpoint_file)


					step+=1
			except tf.errors.OutOfRangeError:
				f.write('Done training for  epochs,steps.\n' )
			finally:
				coord.request_stop()

			coord.join(threads)
			#sess.close()


if __name__=="__main__":
	log_name=str(FLAGS.log_dir)+"alexnet_zooplankton_learningrate_"+'227_227_20'+'_'+re.sub(r'[^0-7]','',str(datetime.datetime.now()))+'.txt'

	f=open(log_name,'w')
	f.write('Alexnet_standard_structure_batch:'+str(FLAGS.batch_size)+'\n')
	f.write('weight_decay,local_response_normalization,learning_rate=0.01\n')
	f.close()
	run_training(log_name)




