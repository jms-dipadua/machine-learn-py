""
# learn_predict.py
# intent:  to make predictions of search relevance based on an input training set
""
# general purpose libraries
import sys
import numpy as np
import pandas as pd
import math 

from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.lda import LDA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

class SearchInput:
	def __init__(self):
		self.get_params()
		self.read_file()

	def get_params(self):
		self.file_dir = "data/"
		self.X_train_file = raw_input("Training Data File (no dir):   ") 
		self.X_test_file = raw_input("Testing Data File (no dir):    ") 

	def read_file(self):
		# get the training data
		X_train_raw = pd.read_csv(self.file_dir + self.X_train_file)
		# we're going to do some sampling to get rid of skew in data 
		# first we'll get the row nums where the relevance is in a range of values
		# <2 (group1); <2.5 & >2 (group2); >2.5 & <3 (group3); == 3 (group4)
		X_train_g1 = X_train_raw.loc[X_train_raw['relevance'] < 2]
		X_train_g2 = X_train_raw.loc[(X_train_raw['relevance'] > 1.9) & (X_train_raw['relevance'] < 2.4)]
		X_train_g3 = X_train_raw.loc[(X_train_raw['relevance'] > 2.6) & (X_train_raw['relevance'] < 3)]
		X_train_g4 = X_train_raw.loc[X_train_raw['relevance'] == 3]
		# THEN we take samples based on those (so our final train data is proportional between the ranges)
		# final samples (w/out replacement)
		X_train_g2_s = X_train_g2.sample(n = X_train_g1.shape[0], replace=False)
		X_train_g3_s = X_train_g3.sample(n = X_train_g1.shape[0], replace=False)
		X_train_g4_s = X_train_g4.sample(n = X_train_g1.shape[0], replace=False)
		# stack them up: this is our final X_train 
		X_train_comp = X_train_g1.append(X_train_g2_s)
		X_train_comp.append(X_train_g3_s)
		X_train_comp.append(X_train_g4_s)
		self.X_train = X_train_comp.drop(['id', 'product_uid', 'relevance'], axis=1)
		self.y_train = X_train_comp['relevance']
		# get the testing data 
		X_test_raw = pd.read_csv(self.file_dir + self.X_test_file)
		self.X_test = X_test_raw.drop(['id', 'product_uid'], axis=1)
		self.fin_df = X_test_raw.drop(['product_uid', 'prod_query_raw_cosine_tfidf', 'prod_query_fixes_cosine_tfidf','des_query_raw_cosine_tfidf','des_query_fixes_cosine_tfidf','kw_matches_overall', 'kw_matches_title', 'kw_matches_des'], axis=1)
	
	def remap_rel(self):
		relevance_map_to_classes = {
			1.00: 1,
			1.33: 1,
			1.67: 2,
			2.00: 3,
			2.33: 4,
			2.5: 4,
			2.50: 4,
			2.67: 5,
			3.00: 6
		}
		
class LearnedPrediction():
	def __init__(self):
		self.search_inputs = SearchInput()
		self.fin_file_name = "data/predictions_v" + raw_input("experiment version number:  ")
		self.pre_process_data()
		self.logit()
		#self.random_forest()
		#self.ann()
		#self.svm()
		#self.ensemble()
		#self.write_file()

	def pre_process_data(self):
		scaler = StandardScaler()
		self.search_inputs.X_train = scaler.fit_transform(self.search_inputs.X_train)
		self.search_inputs.X_test = scaler.fit_transform(self.search_inputs.X_test)

	def svm(self):
		"""
		C_range = np.logspace(-2, 10, 2)
		print C_range
		gamma_range = np.logspace(-9, 3, 2)
		print gamma_range
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = ShuffleSplit(len(self.search_inputs.y_train), n_iter=5, test_size=0.2, random_state=42)
		grid = GridSearchCV(SVR(verbose=True), param_grid=param_grid, cv=cv)
		#grid = GridSearchCV(svm.SVR(kernel='rbf', verbose=True), param_grid=param_grid, cv=cv)
		grid.fit(self.search_inputs.X_train, self.search_inputs.y_train)

		print("The best parameters are %s with a score of %0.2f"
			% (grid.best_params_, grid.best_score_))

		self.svm_preds = grid.predict(self.search_inputs.X_test)
		"""

		regression = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=True)
		regress_fit = regression.fit(self.search_inputs.X_train,self.search_inputs.y_train)
		self.svm_preds = regress_fit.predict(self.search_inputs.X_test)
		
		for i in range(0,len(self.svm_preds) - 1):
			if self.svm_preds[i] < 1:
				self.svm_preds[i] = 1.00
			elif self.svm_preds[i] > 3:
				self.svm_preds[i] = 3.00
		self.search_inputs.fin_df['relevance'] = np.array(self.svm_preds) # easy swap in / out 
		final_file_svm = self.search_inputs.fin_df.to_csv(self.fin_file_name+'_svm.csv', float_format='%.5f', index=False)
		

	def logit(self):
		"""	
		# experiment: create non-continuous "groups"
		#self.y_train = (self.search_inputs.y_train.round() * 2.0 ) / 2.0
		#poly = PolynomialFeatures(3)
		#X_train = poly.fit_transform(self.search_inputs.X_train)
		#X_test = poly.fit_transform(self.search_inputs.X_test)
		"""
		#logit = LinearRegression()
		logit = LogisticRegression()
		#logit = LDA()
		y_train = np.asarray(self.search_inputs.y_train, dtype=str)
		
		logit.fit(self.search_inputs.X_train,self.search_inputs.y_train)
		self.logit_preds = logit.predict(self.search_inputs.X_test)
		self.search_inputs.fin_df['relevance'] = np.array(self.logit_preds) # easy swap in / out 
		final_file_logit = self.search_inputs.fin_df.to_csv(self.fin_file_name+'_logit.csv', float_format='%.5f', index=False)

	def random_forest(self):
		rf = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
		rf.fit(self.search_inputs.X_train, self.search_inputs.y_train)
		self.rf_preds = rf.predict(self.search_inputs.X_test)
		self.search_inputs.fin_df['relevance'] = np.array(self.rf_preds) # easy swap in / out 
		final_file_logit = self.search_inputs.fin_df.to_csv(self.fin_file_name+'_rf.csv', float_format='%.5f', index=False)

	def ann(self):
		#print self.company.X_train.shape[1]
		model = Sequential()
		model.add(Dense(input_dim=self.search_inputs.X_train.shape[1], output_dim=10, init="glorot_uniform"))
		model.add(Activation('tanh'))
		model.add(Dropout(0.1))
		model.add(Dense(input_dim=10, output_dim=10, init="uniform"))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(input_dim=10, output_dim=1, init="glorot_uniform"))
		model.add(Activation("linear"))

		sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_squared_error', optimizer='sgd')
		early_stopping = EarlyStopping(monitor='val_loss', patience=25)
		#epoch_score = model.evaluate(X_score, y_score, batch_size = 16) # this doesn't work
		# first model
		print "fitting first model"
		model.fit(self.search_inputs.X_train, self.search_inputs.y_train, nb_epoch=100, validation_split=.1, batch_size=100, verbose = 1, show_accuracy = True, shuffle = True, callbacks=[early_stopping])
		#score = model.evaluate(self.company.X_cv, self.company.y_cv, show_accuracy=True, batch_size=16)
		self.ann_preds = model.predict(self.search_inputs.X_test)
		#just in case (like w/ svr)
		for i in range(0,len(self.ann_preds) - 1):
			if self.ann_preds[i] < 1:
				self.ann_preds[i] = 1.00
			elif self.ann_preds[i] > 3:
				self.ann_preds[i] = 3.00

		self.search_inputs.fin_df['relevance'] = np.array(self.ann_preds) # easy swap in / out 
		final_file_ann = self.search_inputs.fin_df.to_csv(self.fin_file_name+'_ann.csv', float_format='%.5f', index=False)

	def ensemble(self):
		self.preds_final = (self.svm_preds + self.ann_preds + self.rf_preds) 
		pass 
	
	def write_file(self):
		# for a singleton model, we just make it a dataframe and write it
		self.search_inputs.fin_df['relevance'] = np.array(self.svm_preds) # easy swap in / out 
		print self.search_inputs.fin_df.shape
		final_file = self.search_inputs.fin_df.to_csv(self.fin_file_name+'_ens.csv', float_format='%.2f', index=False)

if __name__ == "__main__":
	predictions = LearnedPrediction() 