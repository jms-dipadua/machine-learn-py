"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: Elena Cuoco
simple starting script, without the use of MNE
Thanks to @author: alexandrebarachant for his wornderful starting script


"""

import numpy as np
import pandas as pd
from glob import glob
import os

# sklearn stuff 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans as mbk
from sklearn.lda import LDA

# scipy signal 
from scipy.signal import butter, lfilter, boxcar

# neural network
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
 
#############function to read data###########

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

    #X_left = np.array([X_train[:,12]])
    #X_left = np.transpose(X_left)

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
    y_train2 = np.hstack((y_train, senventh_label_vals))


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

    #test_left = np.array([X_test[:,12]])
    #test_left = np.transpose(test_left)

    #### BiPolar and Laplacian filters #####


    ################ Train classifiers ########################################
    X_train=data_preprocess_train(X_train)
    X_test=data_preprocess_test(X_test)

    """
    ############# K-CLUSTER ASSIGNMENT ########
    print "TRAINING K-CLUSTER ASSIGNMENT"
    # for sub-sampling 
    X_train = X_train[::subsample, :]

    model = mbk(n_clusters=7, max_iter = 500, max_no_improvement = 15)        
    model.fit(X_train, y_train)
    print model.score(X_train, y_train)
    
    print "Test Cluster Assignment"
    test_preds = []
    for test in X_test:
        pred_shell = [0,0,0,0,0,0,0]
        # Predict the closest cluster each sample in X_test belongs to.
        # effectively an index
        predictions = model.predict(test)
        pred_shell[predictions] = 1     
        test_preds.append(pred_shell)
    
    test_preds = np.array(test_preds)
    #print preds.shape
    X_test = np.hstack((X_test, test_preds))
    #X_test = np.array(test_preds)
    
    print "X-vals Cluster Assignment"
    X_preds = []
    for x_vals in X_train:
        pred_shell = [0,0,0,0,0,0,0]
        x_train_node = model.predict(x_vals)
        pred_shell[x_train_node] = 1    
        X_preds.append(pred_shell)    
    X_train = np.hstack((X_train, X_preds))
    #X_train = np.array(X_preds)

    # memory clean
    test_preds = None 
    X_preds = None
    
    """

    ##########  NN ################
    print('Train subject %d' % (subject))
        
    model = Sequential()
    print "Training 6-class model"
    model.add(Dense(input_dim=X_train.shape[1], output_dim=26, init="glorot_uniform"))
    model.add(Activation("relu"))
    model.add(Dense(input_dim=26, output_dim=10, init="glorot_uniform"))
    model.add(Activation("sigmoid"))
    model.add(Dense(input_dim=10, output_dim=y_train.shape[1], init="glorot_uniform"))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    # first model
    print "fitting first model"
    model.fit(X_train[::subsample,:], y_train[::subsample,:], nb_epoch=80, validation_split=0.15, batch_size=16)

    predictions = model.predict_proba(X_test)
    preds = predictions[:, 0:6]

    # second model
    print "fitting second model" 
    model.fit(X_train[::subsample2,:], y_train[::subsample2,:], nb_epoch=80, validation_split=0.15, batch_size=16)
    predictions = model.predict_proba(X_test)
    preds2 = predictions[:, 0:6]

    preds_a_final = (preds * .67) + (preds2 * .33)
    #preds_final = (preds * .67) + (preds2 * .33)

    print "Training 7-class model"
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], output_dim=26, init="glorot_uniform"))
    model.add(Activation("relu"))
    model.add(Dense(input_dim=26, output_dim=10, init="glorot_uniform"))
    model.add(Activation("sigmoid"))
    model.add(Dense(input_dim=10, output_dim=y_train2.shape[1], init="glorot_uniform"))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    # first model
    print "fitting first model"
    model.fit(X_train[::subsample,:], y_train2[::subsample,:], nb_epoch=80, validation_split=0.15, batch_size=16)

    predictions = model.predict_proba(X_test)
    preds = predictions[:, 0:6]

    # second model 
    print "fitting second model"
    model.fit(X_train[::subsample2,:], y_train2[::subsample2,:], nb_epoch=80, validation_split=0.15, batch_size=16)
    predictions = model.predict_proba(X_test)
    preds2 = predictions[:, 0:6]

    preds_b_final = (preds * .67) + (preds2 * .33)

    preds_final = (preds_a_final + preds_b_final) / 2

    pred_tot.append(preds_final)

# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')