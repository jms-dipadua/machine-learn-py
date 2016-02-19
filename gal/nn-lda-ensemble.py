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
from sklearn.ensemble import RandomForestClassifier

# scipy signal 
from scipy.signal import butter, lfilter, boxcar
from scipy.fftpack import fft, rfft, fftfreq

# neural network
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
 
#############function to read data###########

def butterworth_filter(X,t,k,l):
    if t==0:
        freq=[k, l]
        b,a = butter(3,np.array(freq)/250.0,btype='bandpass')
        X = lfilter(b,a,X)
    elif t==1:
        b,a = butter(3,k/250.0,btype='lowpass')
        X = lfilter(b,a,X)
    elif t==2:
        b,a = butter(3,l/250.0,btype='highpass')
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

def data_preprocess_train(X, p_type):
    #do here your preprocessing
    if p_type == 'frequency':
        scaler= StandardScaler()
        X_prep_normal = np.empty(X.shape)  
        X_prep_normal = rfft(X)
        # scale data
        X_prep_normal = scaler.fit_transform(X_prep_normal)
    elif p_type == 'butter':
        scaler = StandardScaler()
        X_prep_normal = scaler.fit_transform(X)
        X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
        for i in range(10):
            X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
            X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
        X_prep_low_pow = X_prep_low ** 2
        X_prep_normal = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)
    
    return X_prep_normal, scaler

def data_preprocess_test(X, scale, p_type):
    #do here your preprocessing
    if p_type == 'frequency':
        X_prep_normal = np.empty(X.shape)  
        X_prep_normal = rfft(X)
        # scale data
        X_prep_normal = scale.fit_transform(X_prep_normal)
    elif p_type == 'butter':
        X_prep_normal = scale.fit_transform(X)
        X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
        for i in range(10):
            X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
            X_prep_low[:,i] = scale.fit_transform(X_prep_low[:,i])
        X_prep_low_pow = X_prep_low ** 2
        X_prep_normal = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)

    return X_prep_normal

def average_data(data, subsample):
    total_bins = int(data.shape[0] / subsample)
    start = 0
    end = subsample
    step = 1
    new_X = np.empty((total_bins, data.shape[1]))
    for i in range(0, total_bins):
        #get the slice to aggregate accross
        sample_X = data[start:end:step]
        new_X[i,:] = np.mean(sample_X, axis = 0)
        #increment start and end for next slice
        end += subsample
        start += subsample
        #safety switch due to potential rounding
        if end > data.shape[0]:
            break
    # return the new_X
    return new_X


def aggregate_classifications(y_vals, subsample):
    total_bins = int(y_vals.shape[0] / subsample)
    #print total_bins
    start = 0
    end = subsample
    step = 1
    new_y = np.empty((total_bins, y_vals.shape[1]))
    #print new_y.shape
    for i in range(0, total_bins):
        #get the slice to aggregate accross
        sample_y = y_vals[start:end:step]
        new_y[i,:] = np.sum(sample_y, axis = 0)
        #increment start and end for next slice
        end += subsample
        start += subsample
        #safety switch due to potential rounding
        if end > y_vals.shape[0]:
            break
    # get the argmax
    for i in range(0, new_y.shape[0]):        
        #for val in new_y[i,:]: #v1    
        for j in range(len(new_y[i,:])): # v2 corrective stuff (let 'em all be 1s!!)
            if new_y[i,j] > 0:
                new_y[i,j] = 1
            else:
                pass
     
            """
            # v1 corrective stuff
            if val > 1:
                index_max = np.argmax(new_y[i,:])
                index_plus = index_max + 1
                # to avoid observations having multiple categories
                if index_plus == len(new_y[i,:]): 
                    # if this, then simply set last value in the array to 1
                    new_y[i,:] = 0
                    new_y[i,index_max] = 1
                else: # do all the corrective checks
                    if new_y[i,index_max] == new_y[i,index_plus]:
                        # check new_y[i-1,index_max]
                        if new_y[i-1,index_max] == 1:
                            #do something
                            new_y[i,:] = 0
                            new_y[i,index_plus] = 1
                        else:
                            #something else
                            new_y[i,:] = 0
                            new_y[i,index_max] = 1
                    else:
                        new_y[i,:] = 0
                        new_y[i,index_max] = 1
                    break
            """
        
    # return new_y values 
    return new_y


# training subsample.if you want to downsample the training data
subsample = 66 # 66 == 99ms, the typical "binning" for neuroscientists
subsample2 = 130
subsample3 = 40
subsample4 = 25

#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']


#######number of subjects###############
subjects = range(1,2)
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
    """
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
    y_train = np.hstack((y_train, senventh_label_vals))
    """

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

    #### PREP DATA FOR TRAINING #####
    X_train1, scale1 = data_preprocess_train(X_train, 'frequency')
    X_test1 = data_preprocess_test(X_test, scale1, 'frequency')

    X_train2, scale2 = data_preprocess_train(X_train, 'butter')
    X_test2 = data_preprocess_test(X_test, scale2, 'butter')
    
    #X_train_bin = average_data(X_train, subsample)
    #X_test = average_data(X_test, subsample)
    #y_train_bin = aggregate_classifications(y_train, subsample)

    #X_train3 = np.hstack((X_train1, X_train2))
    #X_test3 = np.hstack((X_test1, X_test2))

    ################ Train classifiers ########################################
    

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
    print('Train NN - subject %d' % (subject))
    """
    model = Sequential()
    model.add(Dense(input_dim=X_train1.shape[1], output_dim=int(X_train1.shape[1] * .6), init="glorot_uniform"))
    model.add(Activation("relu"))
    model.add(Dense(input_dim=int(X_train1.shape[1] * .6), output_dim=int(X_train1.shape[1] * .6), init="glorot_uniform"))
    model.add(Activation("sigmoid"))
    model.add(Dense(input_dim=int(X_train1.shape[1] * .6), output_dim=y_train.shape[1], init="glorot_uniform"))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    # first model
    #model.fit(X_train_bin, y_train_bin, nb_epoch=75, validation_split=0.25, batch_size=100, show_accuracy=1)
    model.fit(X_train1[::subsample3,:], y_train[::subsample3], nb_epoch=30, validation_split=0.25, batch_size=50, show_accuracy=1)

    predictions = model.predict_proba(X_test)
    nn_pred1 = predictions[:, 0:6]
    #pred_tot.append(nn_pred1)
    """

    # second model 
    model = Sequential()
    model.add(Dense(input_dim=X_train2.shape[1], output_dim=int(X_train2.shape[1] * .6), init="glorot_uniform"))
    model.add(Activation("relu"))
    model.add(Dense(input_dim=int(X_train2.shape[1] * .6), output_dim=int(X_train2.shape[1] * .6), init="glorot_uniform"))
    model.add(Activation("sigmoid"))
    model.add(Dense(input_dim=int(X_train2.shape[1] * .6), output_dim=y_train.shape[1], init="glorot_uniform"))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    model.fit(X_train2[::subsample3,:], y_train[::subsample3,:], nb_epoch=30, validation_split=0.25, batch_size=50, show_accuracy=1)
    predictions = model.predict_proba(X_test)
    print predictions
    print predictions.shape
    exit()

    nn_pred2 = predictions[:, 0:6]

    preds_final = (nn_pred1 * .50) + (nn_pred2 * .50)
    pred_tot.append(preds_final) # to just do the NN w/ different dimensions

    
    #pred1 = np.empty((X_test.shape[0],6))
    #pred2 = np.empty((X_test.shape[0],6))
    #pred3 = np.empty((X_test.shape[0],6))
    #pred4 = np.empty((X_test.shape[0],6))


    #pred_lr = np.empty((X_test.shape[0],6))
    """
    print "LDA and Logistic Regression"
    lr1 = LDA()
    lr2= LogisticRegression()
    lr3 = LDA()
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion="entropy", random_state=1)
    rf2 = RandomForestClassifier(n_estimators=150, n_jobs=-1, criterion="entropy", random_state=1)

    for i in range(6):
        #lr_y_train = y_train[:,i]
        print('Train subject %d, class %s' % (subject, cols[i]))
        # since i've binned the data, i'm removing the [::subsample,:] parts from X_train, etc
        lr1.fit(X_train2[::subsample3,:],lr_y_train[::subsample3])
        #rf.fit(X_train[::subsample,:],lr_y_train[::subsample]) # change from logistic regression: lr2
        rf.fit(X_train2[::subsample3,:],lr_y_train[::subsample3]) # change from logistic regression: lr2
        lr2.fit(X_train2[::subsample3,:],lr_y_train[::subsample3])
        #lr3.fit(X_train[::subsample3,:],lr_y_train[::subsample3])
        #rf2.fit(X_train[::subsample,:],lr_y_train[::subsample])

        pred1[:,i] = lr1.predict_proba(X_test)[:,1]
        pred2[:,i] = rf.predict_proba(X_test)[:,1] # change from logistic regression lr2
        pred3[:,i] = lr2.predict_proba(X_test)[:,1]
        #pred4[:,i] = lr3.predict_proba(X_test)[:,1]

        #pred_lr[:,i]=pred1[:,i]*0.17 + pred2[:,i]*0.17 + pred3[:,i]*0.33 + pred3[:,i]*0.33
        
    predictions_final = pred1*.19 + pred2*.24 + pred3*.19 + nn_pred1 *.19 + nn_pred2 *.19
    #predictions_final = (pred1 + pred2 + pred3) / 3
    #predictions_final = pred1*.4 + pred2*.3 + pred3*.3
    #predictions_final = (pred2 + nn_pred1) / 2
    #pred_tot.append(predictions_final)
    #pred_tot.append(pred_lr)
    """
    
# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')