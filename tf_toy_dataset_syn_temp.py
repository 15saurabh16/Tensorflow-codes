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
import tf_record_generator as tfg

MIN_AFTER_DEQUEUE = 100000 #100
BATCH_SIZE = 1000



if __name__ == '__main__':
   savedir = '/Users/s.agrawalairan/OneDrive - CRITEO/InternProject/Data/TfSynData/'
   num_samples = 100000
   ind1 = 2
   ind2 = 4
   # SPARSE Synthetic Data 1
   X_a = np.random.randn(num_samples,2)
   X_b = np.zeros((num_samples,9))
#   X = np.concatenate((X_a,X_b),axis=1)
   
   X = np.zeros((num_samples,11))
   X[:,ind1] = X_a[:,0]
   X[:,ind2] = X_a[:,1]
#   Y = np.zeros((num_samples,1))
#   Y[X_a[:,0]+X_a[:,1]<0] = -1
#   Y[X_a[:,0]+X_a[:,1]>=0] = 1
   
   Y = np.zeros((num_samples,1))
   Y[:,0] = X[:,ind1]**2#+X[:,ind2]**3 + 0*np.random.randn(num_samples)
   NonZeroInds = np.random.randint(2,2+9-1,(num_samples,))
   for i in range(num_samples):
       if NonZeroInds[i] != ind1 and NonZeroInds[i] != ind2:
           X[i,NonZeroInds[i]] = 3

   AllZeroInds = np.random.randint(0,num_samples,num_samples/100)
#   Y[AllZeroInds,0] = 0
   X[AllZeroInds,:] = 0
   FinalDataset = np.concatenate((X,Y),axis=1)
   np.savetxt(savedir+"SynDataset1_nonlinear_allzerosamples.csv", FinalDataset, delimiter=",")
      
   # Generating a tf record file
   tfrecord_file = savedir+"SynDataset1_nonlinear_allzerosamples.tfrecords"
   tfg.write_to_tfrecord(X,Y,tfrecord_file)
#   tfg.print_tfrecords_debug(tfrecord_file)
   print "END OF DATASET 1"
   
   ############################################################################
   # SPARSE Synthetic Data 2
   X_a = np.random.randn(num_samples,2)
   X_b = np.zeros((num_samples,9))
#   X = np.concatenate((X_a,X_b),axis=1)
   
   X = np.zeros((num_samples,11))
   X[:,ind1] = X_a[:,0]
   X[:,ind2] = X_a[:,1]
#   Y = np.zeros((num_samples,1))
#   Y[X_a[:,0]+X_a[:,1]<0] = -1
#   Y[X_a[:,0]+X_a[:,1]>=0] = 1
   
   Y = np.zeros((num_samples,1))
   Y[:,0] = X[:,ind1]**2#+X[:,ind2]**3 + 0*np.random.randn(num_samples)
   NonZeroInds = np.random.randint(2,2+9-1,(num_samples,))
   for i in range(num_samples):
       if NonZeroInds[i] != ind1 and NonZeroInds[i] != ind2:
           X[i,NonZeroInds[i]] = 4
   AllZeroInds = np.random.randint(0,num_samples,num_samples/100)
#   Y[AllZeroInds,0] = 0
#   X[AllZeroInds,:] = 0
   FinalDataset = np.concatenate((X,Y),axis=1)
   np.savetxt(savedir+"SynDataset2_nonlinear_allzerosamples.csv", FinalDataset, delimiter=",")
      
   # Generating a tf record file
   tfrecord_file = savedir+"SynDataset2_nonlinear_allzerosamples.tfrecords"
   tfg.write_to_tfrecord(X,Y, tfrecord_file)
   
   ############################################################################
   # SPARSE Synthetic Data 3
   X_a = np.random.randn(num_samples,2)
   X_b = np.zeros((num_samples,9))
#   X = np.concatenate((X_a,X_b),axis=1)
   
   X = np.zeros((num_samples,11))
   X[:,ind1] = X_a[:,0]
   X[:,ind2] = X_a[:,1]
#   Y = np.zeros((num_samples,1))
#   Y[X_a[:,0]+X_a[:,1]<0] = -1
#   Y[X_a[:,0]+X_a[:,1]>=0] = 1
   
   Y = np.zeros((num_samples,1))
   Y[:,0] = X[:,ind1]**2#+X[:,ind2]**3 + 0*np.random.randn(num_samples)
   NonZeroInds = np.random.randint(2,2+9-1,(num_samples,))
   for i in range(num_samples):
       if NonZeroInds[i] != ind1 and NonZeroInds[i] != ind2:
           X[i,NonZeroInds[i]] = 5

   AllZeroInds = np.random.randint(0,num_samples,num_samples/100)
#   Y[AllZeroInds,0] = 0
#   X[AllZeroInds,:] = 0

   FinalDataset = np.concatenate((X,Y),axis=1)
   np.savetxt(savedir+"SynDataset3_nonlinear_allzerosamples.csv", FinalDataset, delimiter=",")
      
   # Generating a tf record file
   tfrecord_file = savedir+"SynDataset3_nonlinear_allzerosamples.tfrecords"
   tfg.write_to_tfrecord(X,Y, tfrecord_file)
   
   ############################################################################
   # SPARSE Synthetic Data 4
   X_a = np.random.randn(num_samples,2)
   X_b = np.zeros((num_samples,9))
   #   X = np.concatenate((X_a,X_b),axis=1)
   
   X = np.zeros((num_samples,11))
   X[:,ind1] = X_a[:,0]
   X[:,ind2] = X_a[:,1]
#   Y = np.zeros((num_samples,1))
#   Y[X_a[:,0]+X_a[:,1]<0] = -1
#   Y[X_a[:,0]+X_a[:,1]>=0] = 1
   
   Y = np.zeros((num_samples,1))
   Y[:,0] = X[:,ind1]**2#+X[:,ind2]**3 + 0*np.random.randn(num_samples)
   NonZeroInds = np.random.randint(2,2+9-1,(num_samples,))
   for i in range(num_samples):
       if NonZeroInds[i] != ind1 and NonZeroInds[i] != ind2:
           X[i,NonZeroInds[i]] = 6

   AllZeroInds = np.random.randint(0,num_samples,num_samples/1000)
#   Y[AllZeroInds,0] = 0
#   X[AllZeroInds,:] = 0

   FinalDataset = np.concatenate((X,Y),axis=1)
   np.savetxt(savedir+"SynDataset4_nonlinear_allzerosamples.csv", FinalDataset, delimiter=",")
      
   # Generating a tf record file
   tfrecord_file = savedir+"SynDataset4_nonlinear_allzerosamples.tfrecords"
   tfg.write_to_tfrecord(X,Y, tfrecord_file)
   
   ############################################################################
   tfg.print_tfrecords_debug(tfrecord_file)

    # Reading one example at a time from tfRecordsfile
#   filename = savedir+"SynDataset4_nonlinear_allzerosamples.tfrecords"
#   for serialized_example in tf.python_io.tf_record_iterator(filename):
#       print "hello"
#       example = tf.train.Example()
#       example.ParseFromString(serialized_example)
#        # traverse the Example format to get data
#       label = example.features.feature['label'].float_list.value
#       ids = example.features.feature['ids'].int64_list.value
#       values = example.features.feature['values'].float_list.value
#       print label, ids, values
       
   # Reading batches of data from multip\e files. 
   