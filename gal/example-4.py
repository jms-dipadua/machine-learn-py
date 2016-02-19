# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: Elena Cuoco
simple starting script, without the use of MNE
Thanks to @author: alexandrebarachant for his wornderful starting script


"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, boxcar
from numpy import convolve
from sklearn.linear_model import LogisticRegression
from glob import glob
import os
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.qda import QDA
from sklearn.decomposition import FastICA, PCA
from sklearn.pipeline import Pipeline as pipe

 
#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

def butterworth_filter(X,t,k,l):
    if t==0:
        freq=[k, l]
        b,a = butter(3,np.array(freq)/500.0,btype='bandpass')
        X = lfilter(b,a,X)
    elif t==1:
        b,a = butter(3,k/500.0,btype='lowpass')
        X = lfilter(b,a,X)
    elif t==2:
        b,a = butter(3,l/500.0,btype='highpass')
        X = lfilter(b,a,X)      
    return X

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data

scaler= StandardScaler()
def data_preprocess_train(X):
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
    for i in range(10):
        X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
        X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
    X_prep_low_pow = X_prep_low ** 2
    X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)
    #do here your preprocessing
    return X_prep

def data_preprocess_test(X):
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
    for i in range(10):
        X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
        X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
    X_prep_low_pow = X_prep_low ** 2
    X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)
    return X_prep

# training subsample.if you want to downsample the training data
subsample  = 70
subsample2 = 130
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

# submission file
submission_file = raw_input("Name final submission file:    ")


#######number of subjects###############
subjects = range(1,2)
ids_tot = []
pred_tot = []

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw= []
    raw = []
    ################ READ DATA ################################################
    fnames =  glob('train2/subj%d_series*_data.csv' % (subject))
    for fname in fnames:
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)

    X = pd.concat(raw)
    y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
    X_train =np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))


    ################ Read test data #####################################
    #
    fnames =  glob('test2/subj%d_series*_data.csv' % (subject))
    test = []
    idx=[]
    for fname in fnames:
      data=prepare_data_test(fname)
      test.append(data)
      idx.append(np.array(data['id']))
    X_test= pd.concat(test)
    ids=np.concatenate(idx)
    ids_tot.append(ids)
    X_test=X_test.drop(['id' ], axis=1)#remove id
    #transform test data in numpy array
    X_test =np.asarray(X_test.astype(float))


    ################ Train classifiers ########################################
    lr1 = LDA()
    lr2=LogisticRegression()
    lr3 = LDA()
   
    ica = FastICA()
    pca = PCA()
    
    pred1 = np.empty((X_test.shape[0],6))
    pred2 = np.empty((X_test.shape[0],6))
    pred3 = np.empty((X_test.shape[0],6))
    

    pred = np.empty((X_test.shape[0],6))
    
    X_train=data_preprocess_train(X_train)
    X_test=data_preprocess_test(X_test)
    for i in range(6):
        y_train= y[:,i]
        print('Train subject %d, class %s' % (subject, cols[i]))
        print "First Pipeline"
        class1 = pipe(steps = [('fica', ica),('lda', lr1)])
        class1.fit(X_train[::subsample,:],y_train[::subsample])
        print "Second Pipeline"
        class2 = pipe(steps = [('fica', ica),('lda', lr2)])
        class2.fit(X_train[::subsample,:],y_train[::subsample])
        print "Third Pipeline"
        class3 = pipe(steps = [('fica', ica),('lda', lr3)])
        class3.fit(X_train[::subsample2,:],y_train[::subsample2])
        #lr1.fit(X_train[::subsample,:],y_train[::subsample])
        #lr2.fit(X_train[::subsample,:],y_train[::subsample])
        #lr3.fit(X_train[::subsample2,:],y_train[::subsample2])
        
        pred1[:,i] = class1.predict_proba(X_test)[:,1]
        pred2[:,i] = class2.predict_proba(X_test)[:,1]
        pred3[:,i] = class3.predict_proba(X_test)[:,1]
        
        pred[:,i]=pred1[:,i]*0.24+pred2[:,i]*0.24+pred3[:,i]*0.52

    pred_tot.append(pred)

# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')