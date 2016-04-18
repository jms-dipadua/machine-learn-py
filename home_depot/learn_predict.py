""
# learn_predict.py
# intent:  to make predictions of search relevance based on an input training set
""
# general purpose libraries
import sys
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV

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
		X_train_g2 = X_train_raw.loc[(X_train_raw['relevance'] > 2) & (X_train_raw['relevance'] < 2.5)]
		X_train_g3 = X_train_raw.loc[(X_train_raw['relevance'] > 2.5) & (X_train_raw['relevance'] < 3)]
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
		self.fin_df = X_test_raw.drop(['product_uid', 'prod_query_raw_cosine_tfidf', 'prod_query_fixes_cosine_tfidf','des_query_raw_cosine_tfidf','des_query_fixes_cosine_tfidf','kw_matches'], axis=1)

class LearnedPrediction():
	def __init__(self):
		self.search_inputs = SearchInput()
		self.fin_file_name = "data/predictions_v" + raw_input("experiment version number:  ") + ".csv"
		self.pre_process_data()
		self.svm()
		#self.logit()
		self.write_file()

	def pre_process_data(self):
		scaler = StandardScaler()
		self.search_inputs.X_train = scaler.fit_transform(self.search_inputs.X_train)
		self.search_inputs.X_test = scaler.fit_transform(self.search_inputs.X_test)

	def svm(self):
		C_range = np.logspace(-2, 10, 3)
		print C_range
		gamma_range = np.logspace(-9, 3, 3)
		print gamma_range
		exit()
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedShuffleSplit(self.search_inputs.y_train, n_iter=5, test_size=0.2, random_state=42)
		grid = GridSearchCV(svm.SVR(kernel='rbf', verbose=True), param_grid=param_grid, cv=cv)
		grid.fit(self.search_inputs.X_train, self.search_inputs.y_train)

		print("The best parameters are %s with a score of %0.2f"
			% (grid.best_params_, grid.best_score_))

		self.svm_preds = grid.predict(self.search_inputs.X_test)

		regression = svm.SVR(kernel='rbf', C=100, gamma=0.1, verbose=True)
		#regress_fit = regression.fit(self.search_inputs.X_train,self.search_inputs.y_train)
		#self.svm_preds = regress_fit.predict(self.search_inputs.X_test)

	def logit(self):
		logit = LogisticRegression()
		logit.fit(self.search_inputs.X_train,self.search_inputs.y_train)
		self.logit_preds = logit.predict(self.search_inputs.X_test)

	def relevance_vote(self):
		pass 
	
	def write_file(self):
		# for a singleton model, we just make it a dataframe and write it
		self.search_inputs.fin_df['relevance'] = np.array(self.svm_preds) # easy swap in / out 
		print self.search_inputs.fin_df.shape
		final_file = self.search_inputs.fin_df.to_csv(self.fin_file_name, float_format='%.2f', index=False)

if __name__ == "__main__":
	predictions = LearnedPrediction() 