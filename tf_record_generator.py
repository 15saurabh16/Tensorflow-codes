#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 22:51:05 2018

@author: s.agrawalairan
"""


import tensorflow as tf
import numpy as np
import os
import pickle as pk
import pdb



# Version to  
def write_to_tfrecord(X,Y,tfrecord_file):
 """ This example is to write a sample to TFRecord file. If you want to write
 more samples, just use a loop.
 """
 writer = tf.python_io.TFRecordWriter(tfrecord_file)
 for i in range(X.shape[0]):
       nz_ids = np.where(X[i,:]!=0)[0]
       nz_entries = X[i,nz_ids]
       example = tf.train.Example(features=tf.train.Features(feature={
       'label': tf.train.Feature(float_list=tf.train.FloatList(value=[Y[i]])),
       'ids': tf.train.Feature(int64_list=tf.train.Int64List(value=nz_ids)),
       'values':tf.train.Feature(float_list=tf.train.FloatList(
       value=nz_entries))
       }))
       writer.write(example.SerializeToString())
 writer.close()
 return