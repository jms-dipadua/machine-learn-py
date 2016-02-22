import numpy as np
import pandas as pd
from glob import glob
import sys 
from collections import defaultdict
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline as pipe
from sklearn import decomposition as decomp
from sklearn import grid_search

# neural network
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from theano import config as t_fig



def read_file(file_name):
    # Read data
    print "reading file data"
    data = pd.read_csv(file_name)
    #read_data = data.convert_objects(convert_numeric=True)

    return data

def split_train_outcomes(X_data):
	print "spliting train and outcomes"
	#print X_data.shape
	#X_vals = X_data.drop([], axis=1)
	X_vals = X_data.drop(['id', 'ID', 'target'], axis=1) 
	#X_vals = X_data.drop(['ID'], axis=1)
	#X_vals = np.delete(X_data, 0, 0)
	#X_vals = np.delete(X_vals, 0, 0)
	
	#total_vals = X_vals.shape[1]
	#y_vals = X_vals[:, total_vals-1]
	y_vals = X_data['target']
	#print y_vals
	#print y_vals.shape

	#X_vals = np.delete(X_vals, total_vals - 1, 0)
	#print X_vals.shape

	return X_vals, y_vals

def clean_test(X_test):
	print "splitting test vals and ids"
	X_test = X_test.drop(['id'], axis=1) 
	ids = X_test['ID']
	X_test = X_test.drop(['ID'], axis=1)

	return X_test, ids

scaler = StandardScaler()
def scale_data(data_raw): # will scale the provided vector of integers
	print "Scaling non-boolean, non-enum integers and floats"
	data_scaled = scaler.fit_transform(data_raw)
	return data_scaled

def initialize():
	print "TRAIN AND PREDICT"
	print "----   ----"
	print "Provide TRAIN FILE:"
	train_data = raw_input("< ")	
	print "----   ----"
	print "provide TEST FILE:"
	test_data = raw_input("< ")
	print "----   ----"
	print "provide FINAL FILE:"
	fin_file = raw_input("< ")

	train_data = read_file(train_data)
	
	X_test = read_file(test_data)
	X_test, ids = clean_test(X_test)
	

	X_vals, y_vals = split_train_outcomes(train_data)
	X_train, X_score, y_train, y_score = train_test_split(X_vals, y_vals, test_size=0.25, random_state=42)
	train_data = None # clean the memory out
	
	cols = ['target', 'blank'] # blank is included as kludge. dropped before file write
	#cols = ['target']
	# scale the data
	#X_vals = scale_data(X_vals)
	X_train = scale_data(X_train)
	X_score = scale_data(X_score)
	X_test = scale_data(X_test)
	
	print "training and so forth:"
		# Initialize SVD
	#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	#svd = decomp.TruncatedSVD()
	#pca = decomp.PCA(parameters)
	"""
	lr1 = LDA()
	lr2= LogisticRegression()
	rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion="entropy", random_state=1)
	"""
	#svm1 = svm.SVC()
	"""
    # Create the pipeline 
	clf = pipe([('pca', pca),
    	('lr2', lr2)])

	n_components = [100, 300, 600, 900]

	estimator = grid_search.GridSearchCV(clf,dict(pca__n_components=n_components))

	print "fitting model"
	estimator.fit(X_train, y_train)

	print("Best score: %0.3f" % estimator.best_score_)
	print("Best parameters set:")
	best_parameters = estimator.best_estimator_.get_params()
	print best_parameters
    # Get best model
	best_model = estimator.best_estimator_
    
    # Fit model with best parameters
	best_model.fit(X_train,y_train)
	pred = estimator.predict_proba(X_test)
	score = best_model.score(X_score, y_score)
	print "Score for CV data set:   %r" % score 
	
	print "LDA()"
	lr1.fit(X_train,y_train)
	score = lr1.score(X_score, y_score)
	print "LDA SCORE:    %r"  % score 
	print "predicting test values"
	pred1 = lr1.predict_proba(X_test)
	
		print "logistic regression"
	#lr2.fit(X_vals, y_vals)
	lr2.fit(X_train, y_train)
	score = lr2.score(X_score, y_score)
	print "score for CV data set:  %r" % score 
	print "predicting test values"
	pred2 = lr2.predict_proba(X_test)
	

	#print "random forest"
	#rf.fit(X_vals_train, y_vals_train)
	#score = rf.score(X_vals_score, y_vals_score)
	#print "score for non-sampled data set:  %r" % score 
	#print raw_input("PRESS ENTER")
	#print "predicting test values"
	#pred3 = rf.predict_proba(X_test)
	

	print "SVM"
	svm1.fit(X_vals, y_vals)
	lr2.fit(X_vals_train, y_vals_train)
	score = lr2.score(X_vals_score, y_vals_score)
	print "score for non-sampled data set:  %r" % score 
	#print raw_input("PRESS ENTER")
	print "predicting test values"
	pred4 = svm1.predict_proba(X_test)
	"""

	#pred_final = (pred1 + pred2 + pred3) / 3
	"""
	# NEURAL NETWORK        
	model = Sequential()
	model.add(Dense(input_dim=X_train.shape[1], output_dim=1200, init="glorot_uniform"))
	model.add(Activation("tanh"))
	model.add(Dropout(0.5))
	#model.add(Dense(input_dim=400, output_dim=100, init="glorot_uniform"))
	#model.add(Activation("sigmoid"))
	model.add(Dense(input_dim=1200, output_dim=50, init="glorot_uniform"))
	model.add(Activation("tanh"))
	model.add(Dropout(0.5))
	model.add(Dense(input_dim=50, output_dim=1, init="glorot_uniform"))
	model.add(Activation("softmax"))

	sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer='sgd')
	early_stopping = EarlyStopping(monitor='val_loss', patience=5)
	#epoch_score = model.evaluate(X_score, y_score, batch_size = 16) # this doesn't work
	# first model
	print "fitting first model"
	model.fit(X_train, y_train, nb_epoch=1, validation_split=0.25, batch_size=16, verbose = 1, callbacks=[early_stopping])
	score = model.evaluate(X_score, y_score, show_accuracy=True, batch_size=16)
	print score

	nn_preds = model.predict_proba(X_test)
	print nn_preds.shape


	nb_samples = X_train.shape[0]
	
	newshape = (nb_samples, 1, nb_features)
	X_train = np.reshape(X_train, newshape).astype(t_fig.floatX)
	#y_train = y_train.astype(t_fig.floatX)
	"""
	nb_features = X_train.shape[1]
	X_train = X_train.reshape(X_train.shape + (1, ));

	# NEURAL NETWORK  - Convolutional    
	model = Sequential()
	model.add(Convolution1D(nb_filter = 24, filter_length = 1, input_dim =X_train.shape[1]))
	model.add(Activation("tanh"))
	model.add(Dropout(0.1)) # some dropout to help w/ overfitting
	model.add(Convolution1D(nb_filter = 48, filter_length= 1, subsample_length= 1))
	model.add(Activation("tanh"))
	#model.add(MaxPooling1D(pool_size=(2, 2)))
	model.add(Convolution1D(nb_filter = 96, filter_length= 1, subsample_length=1))
	model.add(Activation("tanh"))
	model.add(Dropout(0.2))
	# flatten to add dense layers
	model.add(Flatten())
	#model.add(Dense(input_dim=nb_features, output_dim=50))
	model.add(Dense(nb_features))
	model.add(Activation("tanh"))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Activation("tanh"))
	model.add(Dense(1))
	model.add(Activation("tanh"))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer='sgd')
	early_stopping = EarlyStopping(monitor='val_loss', patience=5)

	print "fitting first model"
	model.fit(X_train, y_train, nb_epoch=1, validation_split=0.25, verbose = 1, callbacks=[early_stopping])
	#score = model.evaluate(X_score, y_score, show_accuracy=True)
	#print score

	nn_preds = model.predict(X_test)
	
	#print nn_preds
	#print nn_preds.shape
	
    
	print "creating submission file"
    # SUBMISSION FILE
	submission = pd.DataFrame(index=ids,
                          columns=cols,
                          data=nn_preds)

	submission = submission.drop(['blank'], axis = 1) # kludge. not sure what was wrong...

	#print submission
	# write file
	print "writing final file"
	submission.to_csv(fin_file,index_label='ID',float_format='%.6f')

initialize()