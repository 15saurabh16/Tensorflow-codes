#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 11:54:38 2018

@author: s.agrawalairan
"""

import tensorflow as tf
import numpy as np
import os
import pickle as pk
import csv
import pandas as pd
import logging
import pprint
import pdb
flags = tf.app.flags
FLAGS = flags.FLAGS
logging.basicConfig(level=logging.INFO)
pprint.PrettyPrinter().pprint(FLAGS.__flags)


# Select optimizer
def get_optimizer(optimizer, learning_rate):
  
  if optimizer == "sgd":
    return tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer == "adadelta":
    return tf.train.AdadeltaOptimizer(learning_rate)
  elif optimizer == "adagrad":
    return tf.train.AdagradOptimizer(learning_rate)
  elif optimizer == "adam":
    return tf.train.AdamOptimizer(learning_rate)
  elif optimizer == "ftrl":
    return tf.train.FtrlOptimizer(learning_rate)
  elif optimizer == "rmsprop":
    return tf.train.RMSPropOptimizer(learning_rate)
  else:
    print("Unknow optimizer, exit now")
    exit(1)


def mkdirnotex(filename):
	folder=os.path.dirname(filename)
	if not os.path.exists(folder):
		os.makedirs(folder)
        
        
def apply_activ_fun(activ_fn,input_tnsr):
    if activ_fn == "sigmoid":
        return tf.nn.sigmoid(input_tnsr)
    elif activ_fn == "relu":
        return tf.nn.relu(input_tnsr)
    elif activ_fn =="tanh":
        return tf.nn.tanh(input_tnsr)
    else:
        print ("Unknown activation function, exit now")
        exit(1)  
 
       
def get_loss_fun(loss_fn_str,y_pr,y_obs,reg_loss_tr):
    if loss_fn_str == "poisson_loss":
         return tf.reduce_sum(tf.nn.log_poisson_loss(y_obs,y_pr,name='Poisson_loss'))
    elif loss_fn_str == "lognormal_loss":
        return tf.reduce_mean(tf.square(tf.log(y_pr) - tf.log(tf.add(y_obs,MIN_Y_VAL))))
    elif loss_fn_str == "mse":
        return tf.reduce_mean(tf.square(y_pr - y_obs))
    elif loss_fn_str =="reg_l1_mse":
        return tf.reduce_mean(tf.square(y_pr - y_obs)) + REG_CONSTANT*reg_loss_tr # Regularized MSE    
    else:
        print ("Unknown loss function, exit now")
        exit(1)  
    
def linear_reg_inference(sparse_ids,sparse_values,num_layers):

    W_ls = []
    Bias_ls = []

    # Reset the graph
#    tf.reset_default_graph()

    with tf.variable_scope("linear_reg"):
        
        W_ls.append(tf.get_variable(
                "weights_0", [FEATURE_SIZE,], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))
        
        Bias_ls.append(tf.get_variable(
                "biases_0", [LABEL_SIZE,], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))    
        
    # Regularization Term
    reg_loss = tf.reduce_sum(tf.abs(W_ls[0]))
    
    # Create the neural net graph
    y_est = tf.nn.embedding_lookup_sparse(W_ls[0], sparse_ids, sparse_values, combiner="sum") + Bias_ls[0]   
    return [y_est,reg_loss]


    
def dnn_inference(sparse_ids,sparse_values,layer_sizes,num_layers,activ_fn,keep_prob_dropout): # num_layers include input layer and output layer

    W_ls = []
    Bias_ls = [[]]
    num_samp = tf.shape(sparse_values)
    num_samp = tf.Print(num_samp, data=[num_samp],
                           message='Num Samples in input ')
    
    with tf.variable_scope("dnn_model"):
        
        for i in range(num_layers-1): 
#            print "LAYER IND = {}".format(i)
            W_ls.append(tf.get_variable("weights_"+str(i), [layer_sizes[i], layer_sizes[i+1]], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))
        
            Bias_ls.append(tf.get_variable("biases_"+str(i+1), [layer_sizes[i+1],], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))    
        
    y_est = tf.nn.embedding_lookup_sparse(W_ls[0], sparse_ids, sparse_values, combiner="sum") + Bias_ls[1]
    
    y_est_sz0 =  tf.shape(y_est)    # DEBUG
    y_est_sz0 = tf.Print(y_est_sz0, data=[y_est_sz0],message='y_est_sz0 ====')  # DEBUG

    y_est = apply_activ_fun(activ_fn,y_est)   
    y_est_sz1 =  tf.shape(y_est)  # DEBUG
    y_est_sz1 = tf.Print(y_est_sz0, data=[y_est_sz1],message='y_est_sz1 ====') # DEBUG

#    y_est = tf.nn.relu(tf.nn.embedding_lookup_sparse(W_ls[0], sparse_ids, sparse_values, combiner="sum") + Bias_ls[1])
    
    
    
    # Computing final output
    for i in range(1,num_layers-1):
        y_est = tf.matmul(y_est, W_ls[i])+Bias_ls[i+1]
        if i < num_layers-2:
            y_est = apply_activ_fun(activ_fn,y_est)
        y_est = tf.nn.dropout(y_est, keep_prob_dropout)        
    
    y_est_sz =  tf.shape(y_est)  # DEBUG
    y_est_sz = tf.Print(y_est_sz1, data=[y_est_sz],message='y_est_sz ====')  # DEBUG
    
#    y_est = tf.reshape(y_est,(y_est_sz[0],))
    y_est = tf.reshape(y_est,(num_samp[0]*0 + y_est_sz[0]*1,))
    return y_est



def input_pipeline(filenames, batch_size):
    # Define a `tf.contrib.data.Dataset` for iterating over one epoch of the data.
    dataset = (tf.data.TFRecordDataset(filenames).
                shuffle(buffer_size=MIN_AFTER_DEQUEUE).
                batch(batch_size))

    return dataset.make_initializable_iterator() 

 
def get_features(tfrecords_file,batch_size):
  iterator = input_pipeline(tfrecords_file, batch_size)
  features_obj = iterator.get_next()
  features = tf.parse_example(
                  features_obj,
                  # Defaults are not specified since both keys are required.
                  features={
                   "label": tf.FixedLenFeature([], tf.float32),
                    "ids": tf.VarLenFeature(tf.int64),
                    "values": tf.VarLenFeature(tf.float32),
                    })
  return iterator, features



def main():
  with tf.Session() as sess:
      sess.run(init_op)

      for ep in range(EPOCH_NUMBER):
          # Resets the iterator at the beginning of an epoch.
          sess.run(train_iterator.initializer)
          sess.run(validate_iterator.initializer)
          saver.save(sess, modelsavedir+'DNN_model')

          try:
              while True:
                 
#                batch_val_arr,batch_label_arr = (sess.run([batch_values,batch_labels]))
#                print "HELLLLLO"
#                print "batch_values_arr = {}, batch_label_arr = {}".format(batch_val_arr,batch_label_arr)
                _, step = sess.run([train_op, global_step])
                    
                # Print state while training
                if step % STEPS_TO_VALIDATE == 0:

#                  logging.info("batch_ids are {0},batch_values are {0},batch_labels are {0}".format(sess.run([batch_ids,batch_values,batch_labels])))
                  sess.run([train_op, global_step])
                  loss_val,train_loss_val, vald_loss_val = sess.run([loss,train_loss, vald_loss])
#                  train_loss_val, vald_loss_val,wts_val,bias_val = sess.run([train_loss, vald_loss, wts,biases])
#                  train_loss_val, vald_loss_val,wts_val,bias_val,y_tr,batch_tr = sess.run([train_loss, vald_loss, wts,biases,model_output_tr,batch_labels])

                  logging.info("Step: {} train_loss: {}, vald_loss: {} ".format(step, train_loss_val,vald_loss_val))
#                  logging.info("Step: {}, train_loss: {}, vald_loss: {} wts_val: {} bias_val: {}".format(step, train_loss_val,vald_loss_val,wts_val,bias_val))
#                  logging.info("Step: {}, train_loss: {}, vald_loss: {} wts_val: {} bias_val: {} y_tr:{} batch_tr:{}".format(step, train_loss_val,vald_loss_val,wts_val,bias_val,y_tr,batch_tr))

                if step % 1000 == 0:
                    saver.save(sess,modelsavedir+'DNN_model',global_step = step,write_meta_graph=False)
          except tf.errors.OutOfRangeError:
              pass      
          print('Done training, epoch {} reached'.format(ep))          


if __name__ == '__main__':
   loaddir = '/Users/s.agrawalairan/OneDrive - CRITEO/InternProject/Data/TfSynData/'
   modelsavedir = '/Users/s.agrawalairan/OneDrive - CRITEO/InternProject/DNNOutputSyn/'
   mkdirnotex(modelsavedir)
       
   BATCH_SIZE = 100
   MIN_AFTER_DEQUEUE = BATCH_SIZE*2 #00000 #100

   VALIDATE_BATCH_SIZE = 100 # Size of 
   EPOCH_NUMBER = 50000
   OPTIMIZER = "sgd"
   ACTIVE_FN = 'sigmoid' #sigmoid # tanh
   LOSS_FN = 'mse'#'reg_l1_mse' # 'mse' 'lognormal_loss' 'poisson_loss'
   LOSS_FN_REG = 'mse'#'mse'
   learning_rate = 0.005
   STEPS_TO_VALIDATE = 100
   REG_CONSTANT = 0.1
   KEEP_PROB = 0.92 #_DROPOUT
   MIN_Y_VAL = 0.01 # To avoid Inf loss values in case of lognormal loss
   FEATURE_SIZE = 11
   LABEL_SIZE = 1
   NUM_HID_LAYERS = 2 # number of hidden layers in dnn (no. of weight matrices and biases vectors will be NUM_LAYERS + 1,)
   hidden_layer_sz = [FEATURE_SIZE*3]*NUM_HID_LAYERS
   layer_sizes = [FEATURE_SIZE]+hidden_layer_sz+[LABEL_SIZE]
#   HIDDEN_NODES = [FEATURE_SIZE]*NUM_HID_LAYERS


   # Get all FileNames
   tfr_tr_filenames = [loaddir+"SynDataset1_nonlinear_allzerosamples.tfrecords",loaddir+"SynDataset3_nonlinear_allzerosamples.tfrecords"]
   tfr_vald_filenames = [loaddir+"SynDataset2_nonlinear_allzerosamples.tfrecords",loaddir+"SynDataset4_nonlinear_allzerosamples.tfrecords"]
   
#   tfr_tr_filenames = [loaddir+"SynDataset1_nonlinear.tfrecords",loaddir+"SynDataset3_nonlinear.tfrecords"]
#   tfr_vald_filenames = [loaddir+"SynDataset2_nonlinear.tfrecords",loaddir+"SynDataset4_nonlinear.tfrecords"]
      
   
   # Get a batch of y and X in tr_features
   tf.reset_default_graph()
   train_iterator, train_features = get_features(tfr_tr_filenames, BATCH_SIZE)
   batch_labels = train_features["label"]
   batch_ids = train_features["ids"]
   batch_values = train_features["values"]
   batch_values_len = batch_values.get_shape().as_list()
   batch_labels_len = batch_labels.get_shape().as_list()
   
   # Get a batch of y and X in vald_features
   validate_iterator,validate_features = get_features(tfr_vald_filenames,VALIDATE_BATCH_SIZE)
   validate_batch_labels = validate_features["label"]
   validate_batch_ids = validate_features["ids"]
   validate_batch_values = validate_features["values"]
   
   
   # Model stored in inference function
   reg_loss_tr = 0

   model_output = dnn_inference(batch_ids,batch_values,layer_sizes,NUM_HID_LAYERS+2,ACTIVE_FN,KEEP_PROB)
   
   # Define Loss function    
   loss = get_loss_fun(LOSS_FN,model_output,batch_labels,reg_loss_tr)
   
   # Setting up optimizer, global_step is used to keep track of number of batches seen far
   global_step = tf.Variable(0, name="global_step", trainable=False)
   optimizer = get_optimizer(OPTIMIZER, learning_rate)
   
   # Define train_operation
   train_op = optimizer.minimize(loss, global_step=global_step)
      
   # Variables of model being resused in order to compute validation error
   tf.get_variable_scope().reuse_variables()
   
   
   # Add more evaluation metrics for training data
   model_output_tr = dnn_inference(batch_ids, batch_values,layer_sizes,NUM_HID_LAYERS+2,ACTIVE_FN,KEEP_PROB)
   train_loss = loss
   
   # Add more evaluation metrics for validation data 
   model_output_vald = dnn_inference(validate_batch_ids, validate_batch_values,layer_sizes,NUM_HID_LAYERS+2,ACTIVE_FN,KEEP_PROB)
#   [model_output_vald,_] = linear_reg_inference(validate_batch_ids, validate_batch_values, NUM_HID_LAYERS)
   vald_loss = get_loss_fun(LOSS_FN,model_output_vald,validate_batch_labels,reg_loss_tr)
   
   
   saver = tf.train.Saver()
#   saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
   
   wts = tf.get_variable("dnn_model/weights_0", [layer_sizes[0],layer_sizes[1]], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64))
   biases = tf.get_variable("dnn_model/biases_1", [layer_sizes[1],], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64))

   # Define init_op
   init_op = [tf.local_variables_initializer(), tf.global_variables_initializer()]
   main()
   
   

   
   