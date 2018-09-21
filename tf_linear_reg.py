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




def linear_reg_inference(sparse_ids,sparse_values,hidden_nodes,num_layers):

#    train_sz = np.shape(Xtrain)[0]
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
#    y_est = tf.reduce_sum(tf.nn.embedding_lookup_sparse(W_ls[0], sparse_ids, sparse_values, combiner="sum")) + Bias_ls[0]   
    y_est = tf.nn.embedding_lookup_sparse(W_ls[0], sparse_ids, sparse_values, combiner="sum") + Bias_ls[0]   
    return [y_est,reg_loss]


def dnn_inference(sparse_ids,sparse_values,hidden_nodes,num_layers):

#    train_sz = np.shape(Xtrain)[0]
    W_ls = []
    Bias_ls = []

    # Reset the graph
#    tf.reset_default_graph()

    with tf.variable_scope("dnn_model"):
        
        W_ls.append(tf.get_variable(
                "weights_0", [FEATURE_SIZE, hidden_nodes[0]], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))
        
        Bias_ls.append(tf.get_variable(
                "biases_0", [hidden_nodes[0],], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))    
        
        for i in range(num_layers-1):
            W_ls.append(tf.get_variable(
                "weights_"+str(i+1), [hidden_nodes[i], hidden_nodes[i+1]], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))
            Bias_ls.append(tf.get_variable(
                "biases_"+str(i+1), [hidden_nodes[i+1],], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))
       
        W_ls.append(tf.get_variable(
            "weights_"+str(num_layers), [hidden_nodes[num_layers-1], LABEL_SIZE], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))
                    
        Bias_ls.append(tf.get_variable(
                "biases_"+str(num_layers), [LABEL_SIZE,], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64)))
   
    
    # Create the neural net graph
    y_est = tf.nn.relu(tf.nn.embedding_lookup_sparse(W_ls[0], sparse_ids, sparse_values, combiner="sum") + Bias_ls[0])
    for i in range(num_layers):
        y_est = tf.nn.relu(tf.matmul(y_est, W_ls[i+1])+Bias_ls[i+1])
        
    return y_est




#def sparse_full_connect(sparse_ids,sparse_values,weights_shape,biases_shape,is_train=True):
#    weights = tf.get_variable(
#        "weights", weights_shape, initializer=tf.initializers.zeros())
#    biases = tf.get_variable(
#        "biases", biases_shape, initializer=tf.initializers.zeros())
#    return tf.nn.embedding_lookup_sparse(
#        weights, sparse_ids, sparse_values, combiner="sum") + biases


## TO BE CHANGED
#def logistic_regression_inference(sparse_ids, sparse_values, is_train=True):
#	with tf.variable_scope("logistic_regression"):
#	  layer = sparse_full_connect(sparse_ids, sparse_values,[FEATURE_SIZE, LABEL_SIZE], [LABEL_SIZE])
#	return layer
#
#
## TO BE CHANGED
#def inference(sparse_ids, sparse_values, is_train=True):
#      return logistic_regression_inference(sparse_ids, sparse_values, is_train)




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

      for _ in range(EPOCH_NUMBER):
          # Resets the iterator at the beginning of an epoch.
          sess.run(train_iterator.initializer)
          sess.run(validate_iterator.initializer)
          try:
              while True:
                _, step = sess.run([train_op, global_step])

                # Print state while training
                if step % STEPS_TO_VALIDATE == 0:

#                  logging.info("batch_ids are {0},batch_values are {0},batch_labels are {0}".format(sess.run([batch_ids,batch_values,batch_labels])))
#                  logging.info("valid_batch_ids are {0},valid_batch_labels are {0}".format(sess.run([validate_batch_values,validate_batch_labels])))

                  train_loss_val, vald_loss_val,wts_val,bias_val,model_output = sess.run([train_loss, vald_loss, wts,biases,model_output_tr])
                  print "Just above logging info line"
                  logging.info("Step: {}, train_loss: {}, vald_loss: {} wts_val: {} bias_val: {} y_tr:{}".format(step, train_loss_val,vald_loss_val,wts_val,bias_val,model_output))
          except tf.errors.OutOfRangeError:
              pass      
          print('Done training, epoch reached')          


if __name__ == '__main__':
   loaddir = '/Users/s.agrawalairan/OneDrive - CRITEO/InternProject/Data/TfSynData/'
   MIN_AFTER_DEQUEUE = 20 #00000 #100
   BATCH_SIZE = 100
   VALIDATE_BATCH_SIZE = 100 # Size of 
   EPOCH_NUMBER = 500
   OPTIMIZER = "sgd"
   learning_rate = 0.01
   STEPS_TO_VALIDATE = 5
   REG_CONSTANT = 0
   
   MIN_Y_VAL = 0.01 # For lognormal loss
   FEATURE_SIZE = 11
   LABEL_SIZE = 1
   NUM_LAYERS = 3 # number of hidden layers in dnn (no. of weight matrices and biases vectors will be NUM_LAYERS + 1,)
   HIDDEN_NODES = [FEATURE_SIZE]*NUM_LAYERS
   # Get all FileNames
   tfr_tr_filenames = [loaddir+"SynDataset1.tfrecords",loaddir+"SynDataset3.tfrecords"]
   tfr_vald_filenames = [loaddir+"SynDataset2.tfrecords",loaddir+"SynDataset4.tfrecords"]
      
   tf.reset_default_graph()

   # Get a batch of y and X in tr_features
   train_iterator, train_features = get_features(tfr_tr_filenames, BATCH_SIZE)
   batch_labels = train_features["label"]
   batch_ids = train_features["ids"]
   batch_values = train_features["values"]

   
   # Get a batch of y and X in vald_features
   validate_iterator,validate_features = get_features(tfr_vald_filenames,VALIDATE_BATCH_SIZE)
   validate_batch_labels = validate_features["label"]
   validate_batch_ids = validate_features["ids"]
   validate_batch_values = validate_features["values"]
   
   
   # Model stored in inference function
#   model_output_tr = dnn_inference(batch_ids,batch_values,HIDDEN_NODES,NUM_LAYERS)
   [model_output_tr,reg_loss_tr] = linear_reg_inference(batch_ids,batch_values,HIDDEN_NODES,NUM_LAYERS)
   
   # Define Loss function 
#   loss = tf.reduce_sum(tf.nn.log_poisson_loss(batch_labels,model_output,name='Poisson_loss')) # Poisson Loss
#   loss = tf.reduce_mean(tf.square(tf.log(model_output_tr) - tf.log(tf.add(batch_labels,MIN_Y_VAL)))) # Log Normal Loss   
   loss = tf.reduce_mean(tf.square(model_output_tr - batch_labels)) + REG_CONSTANT*reg_loss_tr # Regularized MSE    

   
   # Setting up optimizer, global_step is used to keep track of number of batches seen far
   global_step = tf.Variable(0, name="global_step", trainable=False)
   optimizer = get_optimizer(OPTIMIZER, learning_rate)
   
   # Define train_operation
   train_op = optimizer.minimize(loss, global_step=global_step)
      
   # Variables of model being resused in order to compute training and validation error
   tf.get_variable_scope().reuse_variables()
   
   
   # Add more evaluation metrics for training data (TO BE CHANGED...)
#   model_output_tr = dnn_inference(batch_ids, batch_values, HIDDEN_NODES, NUM_LAYERS)
   [model_output_tr,_] = linear_reg_inference(batch_ids, batch_values, HIDDEN_NODES, NUM_LAYERS)
   train_loss = tf.reduce_mean(tf.square(model_output_tr - tf.add(batch_labels,MIN_Y_VAL))) # MSE 
   wts = tf.get_variable("linear_reg/weights_0", [FEATURE_SIZE,], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64))
   biases = tf.get_variable("linear_reg/biases_0", [LABEL_SIZE,], initializer=tf.initializers.random_uniform(minval=-1,maxval=1,dtype=tf.float64))
   
   # Add more evaluation metrixs for testing data (TO BE CHANGED...)
#   model_output_vald = dnn_inference(batch_ids,batch_values,HIDDEN_NODES,NUM_LAYERS)
   [model_output_vald,_] = linear_reg_inference(validate_batch_ids, validate_batch_values, HIDDEN_NODES, NUM_LAYERS)
   vald_loss = tf.reduce_mean(tf.square(model_output_vald - tf.add(validate_batch_labels,MIN_Y_VAL))) #MSE

    # Define init_op
   init_op = [tf.local_variables_initializer(), tf.global_variables_initializer()]
   main()
   
   

   
   