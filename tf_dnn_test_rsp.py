#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:49:10 2018

@author: s.agrawalairan
"""

import tensorflow as tf
import glob
import pickle as pk
import os
import numpy as np
import tf_dnn_train_rsp as tr_rsp
import get_usr_RS_scores as guRS
#from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
import time
# Only for testing, model is not learnt here in this module. requires pre-learnt weights and biases as input
#import operator

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def get_bin_strs(Bin_lower,Bin_upper):   
    BinStrs = []
    for i in range(len(Bin_lower)):
        BinStrs.append('['+str(Bin_lower[i])+'-'+str(Bin_upper[i])+']')
    return BinStrs

def get_count_vec(InputArr,Bin_lower,Bin_upper):    
    CountVec = []
    for i in range(len(Bin_lower)):
        tempvec = InputArr[(InputArr>Bin_lower[i]) & (InputArr<=Bin_upper[i])]
        CountVec.append(tempvec.size)
    return CountVec


def input_pipeline2(filenames,batch_size):
    # Define a `tf.contrib.data.Dataset` for iterating over one epoch of the data.
    dataset = (tf.data.TFRecordDataset(filenames).batch(batch_size))
    return dataset.make_initializable_iterator() 

 
def get_features2(tfrecords_file,batch_size):
  iterator = input_pipeline2(tfrecords_file,batch_size)
  features_obj = iterator.get_next()
  print "INSIDE GET_FEATURES2"
  features = tf.parse_example(
                  features_obj,
                  # Defaults are not specified since both keys are required.
                  features={
                   "label": tf.FixedLenFeature([], tf.float32),
                    "ids": tf.VarLenFeature(tf.int64),
                    "values": tf.VarLenFeature(tf.float32),
                    "user_id":tf.FixedLenFeature([], tf.int64),
                    })
  return iterator, features



def dnn_inference_testing(W_ls,Bias_ls,sparse_ids,sparse_values,num_layers): # num_layers include input layer and output layer

    num_samp = tf.shape(sparse_values)
        
    y_est = tf.nn.embedding_lookup_sparse(W_ls[0], sparse_ids, sparse_values, combiner="sum") + Bias_ls[1]
    y_est = tr_rsp.apply_activ_fun(ACTIVE_FN,y_est)       
  
    # Computing final output
    for i in range(1,num_layers-1):
        y_est = tf.matmul(y_est, W_ls[i])+Bias_ls[i+1]
        if i < num_layers-2:
            y_est = tr_rsp.apply_activ_fun(ACTIVE_FN,y_est)
#        y_est = tf.nn.dropout(y_est,KEEP_PROB)        
    
    y_est = tf.reshape(y_est,(num_samp[0],))
    return y_est
       
          


if __name__ == '__main__':
   tf.reset_default_graph()
   ####INPUT DATA##############################################################
   seedID = 2669 #2009 #0000 IS TRIAL #2009
   NumSeedU = 25000
   NumNonSeedU = 25000
   win_len = 15
   ST_YDAY = 45
   train_len = 60
   test_len = 15
   test_len = 15
   max_len = 90 
   
   # Action Wts
   EvWtDict = {}
   EvWtDict[0] = 1 # view
   EvWtDict[1] = 2 #5 # basket
   EvWtDict[2] = 1000 #10  #sale
        
   # RelWts Parameters
   min_score = 0
   min_th = 0.01
   file_sfx1 = 'SeedID{}_NumSeed_{}_NumNonSeedU_{}_trainlen_{}winlen_{}_STDAY_{}'.format(seedID,NumSeedU,NumNonSeedU,max_len,win_len,ST_YDAY)
   file_sfx_rel = 'RelWts_minscore_{}_minth_{}'.format(min_score,min_th)
   file_sfx_act = 'ActWts_view_{}_basket_{}_sale_{}'.format(EvWtDict[0],EvWtDict[1],EvWtDict[2])
   loaddir = '/Users/s.agrawalairan/OneDrive - CRITEO/InternProject/Data/DataSetVersions/'+file_sfx1+file_sfx_rel+file_sfx_act+'/' 
    
   

    # EXPERIMENTAL PARAMETERS   
   metadatafile = 'CatRSP_UsrTimelineFeatInfo' +file_sfx1 + file_sfx_rel + file_sfx_act
   with open(loaddir+metadatafile+'.pkl') as f:  
      [var_info,AllUsrInds,CatRelWts,FEATURE_SIZE] = pk.load(f)
       
   
   loaddir_test = loaddir + 'TFRecords_Test/'
#   loaddir_test = loaddir + 'TFRecords_Test_copy/'
   tfr_test_prefix = 'CatRSP_UsrTimelineFeatMat_test_'+file_sfx1 + file_sfx_rel + file_sfx_act
   tfr_test_filenames = glob.glob(loaddir_test+tfr_test_prefix+'*') 
   
   # RS Score File Parameters
   scorefile_prfx = 'RSScores_test_'


   # Plot File Prefix
   plt_file_prfx  = 'AUCPlot_test_'
   plt_file_prfx2 = 'PredRSScore_barplot_testing_'
   
   
   LABEL_SIZE = 1
   MIN_Y_VAL = 1 # To avoid Inf loss values in case of lognormal loss
   PARAM_STR1 = '_min_y_val_{}'.format(MIN_Y_VAL)
   

   # BATCHING PARAMETERS
   BATCH_SIZE = 100
   MIN_AFTER_DEQUEUE = BATCH_SIZE*100 #00000 #100
   BATCH_STR = '_batchsz_{}_madeq_{}_'.format(BATCH_SIZE,MIN_AFTER_DEQUEUE)
   
   # THIS FILE PARAMETERS
   VALIDATE_BATCH_SIZE = 1000 # Size of 
   MIN_AFTER_DEQUEUE_TEST = VALIDATE_BATCH_SIZE*20 # Size of 
   res_file_prfx = 'Results_test_'

   # MODEL PARAMETERS
   OPTIMIZER = "sgd"#"adadelta"#"adagrad"#"ftrl"#"sgd"
   ACTIVE_FN = 'sigmoid' #sigmoid # tanh
   LOSS_FN = 'lognormal_loss'#'lognormal_loss'#'mse'# 'lognormal_loss' # 'mse' 'lognormal_loss' 'poisson_loss'
   LOSS_FN_TEST = 'lognormal_loss' #'lognormal_loss' #'mse'
   learning_rate = 0.05
   KEEP_PROB = 0.92 #_DROPOUT
   LAYER_SZ = 100 #FEATURE_SIZE
   NUM_HID_LAYERS = 1 # number of hidden layers in dnn 
   #NOTE: No. of weight matrices and biases vectors will be NUM_LAYERS + 1
   
   # MODEL OUTPUT DIRECTORIES AND FILENAME SUBSTRINGS
   MODEL_STR = '_optim_{}_activefn_{}_lossfn_{}_keepprob_{}_numhidlayers_{}_hidlayersz_{}_learning_rate_{}'.format(OPTIMIZER,ACTIVE_FN,LOSS_FN,KEEP_PROB,NUM_HID_LAYERS,LAYER_SZ,learning_rate)
   MODELFILESTR = BATCH_STR+MODEL_STR+PARAM_STR1
   MODELSAVEDIR = '/Users/s.agrawalairan/OneDrive - CRITEO/InternProject/DNNOutputRSP/'+file_sfx1 + file_sfx_rel + file_sfx_act + '/' + MODELFILESTR + '/'
   RESULTSDIR = MODELSAVEDIR + 'Results'+ '/'
   tr_rsp.mkdirnotex(RESULTSDIR)

   # OTHER EXPERIMENTAL SETUP PARAMETERS
   EPOCH_NUMBER = 500
   STEPS_TO_VALIDATE = 1000
   REG_CONSTANT = 1.0 # Regularization constant 
   
   
   # LAYER SIZES
   hidden_layer_szs = [LAYER_SZ]*NUM_HID_LAYERS
   layer_sizes = [FEATURE_SIZE]+hidden_layer_szs+[LABEL_SIZE]
   
   # WAITING TIME BETWEEN TWO COMPUTATIONS
   sleep_time = 300 # in seconds
   
   #################END OF INPUT PARAMETERS######################################################################################
   while True:
   
       # Defining batch variables and iterators
       testing_iterator,testing_features = tr_rsp.get_features(tfr_test_filenames,VALIDATE_BATCH_SIZE,MIN_AFTER_DEQUEUE_TEST)
       testing_batch_labels = testing_features["label"]
       testing_batch_ids = testing_features["ids"]
       testing_batch_values = testing_features["values"]
       
       # Defining batch variables and iterators without shuffling
       batch_size2 = (test_len - win_len+1)*1000
       test_iter2,test_features2 = get_features2(tfr_test_filenames,batch_size2)
       test_usrids_2 = test_features2['user_id']
       test_batch_ids_2 = test_features2["ids"]
       test_batch_values_2 = test_features2["values"]
       ############################################################################  
    
       # LOADING EXISTING MODEL
#       tf.reset_default_graph() # Resetting default graph
       W_ls_rs = []
       W_tnsor = []
       Bias_tnsor = [[]]
       W_ls_2 = []
        
       Y_est = []
       Y_est_2 = []
       AllTestUsrInds = []
       AllTestLosses = []
       with tf.Session() as sess: 
           print "INSIDE SESSION"
           # LOADING EXISTING MODEL 
           MetaFileLs = glob.glob(MODELSAVEDIR+'DNN_model'+MODELFILESTR+'*.meta')
           saver = tf.train.import_meta_graph(MetaFileLs[-1])
           saver.restore(sess,tf.train.latest_checkpoint(MODELSAVEDIR))
           graph = tf.get_default_graph()
           sess.run(testing_iterator.initializer)
           for i in range(NUM_HID_LAYERS+1):
               W_tnsor.append(graph.get_tensor_by_name('dnn_model/weights_'+str(i)+':0')) 
               Bias_tnsor.append(graph.get_tensor_by_name('dnn_model/biases_'+str(i+1)+':0')) 
           
            
           y_test = dnn_inference_testing(W_tnsor,Bias_tnsor,testing_batch_ids,testing_batch_values,NUM_HID_LAYERS+2)
           test_loss = tr_rsp.get_loss_fun(LOSS_FN_TEST,y_test,testing_batch_labels,0,REG_CONSTANT,MIN_Y_VAL)
           
           # Computing losses over different batches of validation/ testing data
           try:
              while True:
                  test_loss_val,y_test_val,y_orig_sample,y_test_sample = sess.run([test_loss,y_test,testing_batch_labels,y_test])
                  Y_est.append(y_test_val)
                  AllTestLosses.append(test_loss_val)
           except tf.errors.OutOfRangeError:
               pass
           
          # Obtain Predicted RS Scores for all samples in each file without shuffling
           sess.run(test_iter2.initializer)
           y_test_2 = dnn_inference_testing(W_tnsor,Bias_tnsor,test_batch_ids_2,test_batch_values_2,NUM_HID_LAYERS+2)
           try:
              while True:
                  y_test_val_2,test_usrids2_values = sess.run([y_test_2,test_usrids_2])
                  Y_est_2.append(list(y_test_val_2))
                  AllTestUsrInds.append(list(test_usrids2_values))
           except tf.errors.OutOfRangeError:
               pass
    
          
       # Computng Mean and Std Deviation of losses across all batches     
       AllTestLosses = np.array(AllTestLosses)
       MeanLoss = np.mean(AllTestLosses)
       StdDevLoss = np.std(AllTestLosses)
    
       ############################################################################
       ##AUC EVALUATION FOR DIFFERNT THRESHOLDS####################################
       ############################################################################ 
       
       
       # STEP 1: Loading Ground Truth for RS Scores of users from all files
       AllFileUsrScores = {}    
       ScoreFileLs = glob.glob(loaddir_test+scorefile_prfx+'*')
       counter = 0
       for scorefile in ScoreFileLs:
           with open(scorefile) as f:  # Python 3: open(..., 'wb')
               counter+=1
               [AllUsrScores] = pk.load(f)
           AllFileUsrScores = merge_two_dicts(AllFileUsrScores,AllUsrScores)
           
       # Analysis of Observed RS scores
#       maxval_key = max(AllFileUsrScores.iteritems(), key=operator.itemgetter(1))[0]

    
       # STEP 2: Computing Predicted RS Scores of all users (from all files)
       t_beg = time.time()
       AllTestUsrInds = sum(AllTestUsrInds,[])         
       Y_est_2 = sum(Y_est_2,[])
       if LOSS_FN_TEST=='lognormal_loss':
           Y_est_2 = list(np.exp(Y_est_2))
       PredUsrScores = guRS.get_usr_RS_scores(AllTestUsrInds,Y_est_2)
       t_end = time.time()
       print "Predicted RS Scores computed for all users. Time elapsed: {}".format(t_end-t_beg)
       
       # Dictionaries to vectors (Assuming the user indices are in interval [0,NumUsrs]
       NumUsrs = len(AllFileUsrScores)
       RSScoresObs = np.zeros((NumUsrs,))
       RSScoresPred = np.zeros((NumUsrs,))
       for u in AllFileUsrScores.keys():       
           RSScoresObs[u] = AllFileUsrScores[u]
           RSScoresPred[u] = PredUsrScores[u]      
#       RSScoresPred = RSScoresPred/np.max(RSScoresPred)
#       RSScoresObs = RSScoresObs/np.max(RSScoresObs)
       
       # STEP 3: Compute AUC at  diferent thresholds on ground truth scores
       TopFracs = np.arange(0.01,0.3,0.002)
       AllAUCs = np.zeros(np.shape(TopFracs))
       AllScoreTh = np.zeros(np.shape(TopFracs))    
       for i in range(TopFracs.size):
           mu = TopFracs[i]
           score_th = np.percentile(RSScoresObs,(1-mu)*100)
           AllScoreTh[i] = score_th
           UsrLabels = np.zeros(RSScoresObs.shape)
           UsrLabels[RSScoresObs>score_th] = 1
           AllAUCs[i] = metrics.roc_auc_score(UsrLabels,RSScoresPred, average='macro', sample_weight=None)
    
    
       
       # PLOT AUCs vs TopFracs
       plt_filename = plt_file_prfx + MODELFILESTR
       plt.plot(TopFracs,AllAUCs)
#       best_ind = np.where(AllAUCs==np.max(AllAUCs))[0]
#       BestAUC = AllAUCs[best_ind]
#       BestFrac = TopFracs[best_ind]
       plt.xlabel('\mu: Proportion of Users with best true RS scores')
       plt.ylabel('AUC')
#       plt.title('Best AUC = {} ,Best frac = {}'.format(BestAUC,BestFrac))
       plt.title('Prediction Loss: Mean = {} ,Std. Dev = {}'.format(np.round(MeanLoss,2),np.round(StdDevLoss,3)))
       plt.savefig(RESULTSDIR + plt_filename +'.jpg')
       plt.close()
       
       # Stats on RSScoresPred
       plt_filename2 = plt_file_prfx2
       diff= 0.1
       Bin_lower = list(np.round(np.arange(0,2,diff),2)) 
#       Bin_lower = [x +  MIN_Y_VAL for x in Bin_lower]      
       Bin_upper = list(np.round(np.arange(Bin_lower[1],Bin_lower[-1]+2*diff,diff),2)) 
#       Bin_upper = [0.1,0.2,0.5,1,2,5,10,100]
#       Bin_upper = [x +  MIN_Y_VAL for x in Bin_upper]
       BinStrs = get_bin_strs(Bin_lower,Bin_upper)
       NzRSScorePredTest = np.array(filter(lambda a:(a!=0),RSScoresPred))
   
       # BAR PLOT FOR TESTING DATA          
       CountVecTest = get_count_vec(NzRSScorePredTest,Bin_lower,Bin_upper)
       NumZTest = -len(NzRSScorePredTest) +len(RSScoresPred)
       plt.bar(np.arange(len(Bin_lower)), CountVecTest, align='center', alpha=0.5)
       plt.xticks(np.arange(len(Bin_lower)), BinStrs,rotation=60)
       plt.title('Testing: Num Zeros = {}'.format(NumZTest))
       plt.savefig(RESULTSDIR + plt_filename2 + '.jpg')
       plt.close()  
       
#       plt.hist(RSScoresObs,bins=100,range=[0.01,1])
#       plt.xlim([0,1])
#       plt.title('Max:{} Median = {} ,Std. Dev = {}'.format(np.max(RSScoresObs),np.median(RSScoresObs),np.std(RSScoresObs)))
#       plt.hist(RSScoresObs, normed=True, cumulative=True, label='CDF', histtype='step', alpha=0.8, color='k')
#       plt.savefig(RESULTSDIR + plt_filename2)
#       plt.close()

       
       # Saving the results
       res_filename = res_file_prfx + MODELFILESTR       
       with open(RESULTSDIR + res_filename +'.pkl', 'w') as f:  # Python 3: open(..., 'wb')
           pk.dump([AllTestLosses,MeanLoss,StdDevLoss],f)
       print "Sleeping for {} seconds".format(sleep_time)    
       time.sleep(sleep_time)
   

   

