
"""
kclust.py
Grasp, Lift, Release @ Kaggle
__author__ : jms. 

"""

# data loading and general matrix stuff
import numpy as np
import pandas as pd
from glob import glob
import sys 
from collections import defaultdict
import csv

# implementation specific stuff 
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans as mbk
from scipy.signal import butter, lfilter, convolve, boxcar
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, metrics, grid_search
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.decomposition import FastICA, PCA
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline as pipe

# scipy signal 
from scipy.signal import butter, lfilter, boxcar


# outline 
# 1. glob data into a single instance 
	# - we don't care about "subject" beyond its index for the prediction
# 2. send full (training) data set to the k-means clustring: 
	# likely using MiniMatchKMeans due to scalability
	# model = mbk(train_glob, train_y_glob)
# 3. send a glob of the test data to the model:
	# since we cannot use the "future" we'll just go through each data point 
	# preds = model.predict(test_glob[i])

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

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data

scaler= StandardScaler()
def data_preprocess_train(X):
    #do here your preprocessing
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
    for i in range(10):
        X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
        X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
    X_prep_low_pow = X_prep_low ** 2
    X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)
    
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
subsample = 66 # normally to 66 :: 5 for tight data set
subsample2 = 130

#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']


#######number of subjects###############
subjects = range(1,13)
ids_tot = []
pred_tot = []

#### submission file ####
submission_file = raw_input("NAME FINAL FILE:    ")

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    print "STARTING SUBJECT %r" % subject
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
    y_train = np.asarray(y.astype(float))

    # add in the "rest / no action" col for when all vals in y == 0
    senventh_label_vals = []
    for y_val in y_train:
        if not 1 in y_val:
            senventh_label_vals.append(1)
        else:
            senventh_label_vals.append(0)

    senventh_label_vals = np.array([senventh_label_vals])
    senventh_label_vals = np.transpose(senventh_label_vals)
    #print senventh_label_vals.shape


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
    X_train=data_preprocess_train(X_train)
    X_test=data_preprocess_test(X_test)

    mbk_model = mbk(n_clusters=7, max_iter = 1000, max_no_improvement = 15)

    ica = FastICA()
    pca = PCA()
    
    pred1 = np.empty((X_test.shape[0],6)) # LDA NEEDS THIS
    pred2 = np.empty((X_test.shape[0],6)) # LDA NEEDS THIS
    pred3 = [] #np.empty((X_test.shape[0],6))

    
    ############# K-CLUSTER ASSIGNMENT ########
    print "TRAINING K-CLUSTER ASSIGNMENT"

    """
    print "Test Cluster Assignment: Model 1"
    model1 = pipe(steps = [('fica', ica),('mbk_model', mbk_model)])
    model1.fit(X_train[::subsample, :], y_train[::subsample, :])
    print model1.score(X_train[::subsample, :], y_train[::subsample, :])
    
    #for i in range(len(X_test)):
    for test in X_test:
        pred_shell = [0,0,0,0,0,0,0]
        # Predict the closest cluster each sample in X_test belongs to.
        # effectively an index
        prediction = model1.predict(test)
        pred_shell[prediction] = 1     
        pred1.append(pred_shell)
    
    pred1 = np.array(pred1)
    #print pred1
    #print pred1.shape
    #raw_input("PRESS ENTER TO CONTINUE")

    print "Test Cluster Assignment: Model 2"
    model2 = pipe(steps = [('pca', pca),('mbk_model', mbk_model)])
    model2.fit(X_train[::subsample, :], y_train[::subsample, :])
    print model2.score(X_train[::subsample, :], y_train[::subsample, :])
   
    for test in X_test:
        pred_shell = [0,0,0,0,0,0,0]
        # Predict the closest cluster each sample in X_test belongs to.
        # effectively an index
        prediction = model2.predict(test)
        pred_shell[prediction] = 1     
        pred2.append(pred_shell)
    pred2 = np.array(pred2)
    """

    print "Test Cluster Assignment: Model 3"
    model3 = pipe(steps = [('ica', ica),('mbk_model', mbk_model)])
    model3.fit(X_train[::subsample, :], y_train[::subsample, :])
    print model3.score(X_train[::subsample2, :], y_train[::subsample2, :])


    for test in X_test:
        pred_shell = [0,0,0,0,0,0,0]
        # Predict the closest cluster each sample in X_test belongs to.
        # effectively an index
        prediction = model3.predict(test)
        pred_shell[prediction] = 1     
        pred3.append(pred_shell)
    pred3 = np.array(pred3)
    pred3_final = pred3[:, 0:6]


    #pred_agg = pred1*.24 + pred2 * .24 + pred3 *.52

    lr1 = LDA()
    lr2= LogisticRegression()
    lr3 = LDA()
    for i in range(6):
        lr_y_train= y_train[:,i]

        lr1.fit(X_train[::subsample,:],lr_y_train[::subsample])
        #lr2.fit(X_train[::subsample,:],lr_y_train[::subsample])
        lr3.fit(X_train[::subsample2,:],lr_y_train[::subsample2])

        pred1[:,i] = lr1.predict_proba(X_test)[:,1]
        #pred2[:,i] = lr2.predict_proba(X_test)[:,1]
        pred2[:,i] = lr3.predict_proba(X_test)[:,1]

    pred_final=pred1*0.26+pred2*0.48+pred3_final*0.26
    #pred_final = pred_agg[:, 0:6]

    pred_tot.append(pred_final)


# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')