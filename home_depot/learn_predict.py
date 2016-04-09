""
# learn_predict.py
# intent:  to make predictions of search relevance based on an input training set
# why: we can isolate the pipeline by separating transformation from actual agorithmic learning
# plus: we won't have to waste computation cycles (and time) by re-doing steming, etc every time we wnat to run our models
# i.e. we can measure twice and cut once (or revisit the raw if need be)
# input: raw data :: train and test.csv
# output:  ml-ready features. current: TFIDF COSINE Sim Scores 
""
# general purpose libraries
import sys
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.learning_curve import learning_curve

class SearchInput:
	def __init__(self):
		self.get_params()
		self.read_file()

	def get_params(self):
		self.file_dir = "data/"
		self.X_train_file = raw_input("Training Data File (no dir)") 
		self.X_test_file = raw_input("Testing Data File (no dir)") 

	def read_file(self):
		# get the training data
		X_train_test = pd.read_csv(self.file_dir + self.X_train_file)
		self.X_train = X_train_test['cosine_tfidf']
		self.y_train = X_train_test['relevance']
		# get the testing data 
		self.X_test = pd.read_csv(self.file_dir + self.X_test_file)
		self.fin_df = self.X_test.drop(['id', 'cosine_tfidf'], axis=1)

class LearnedPrediction():
	def __init__(self):
		self.search_inputs = SearchInput()
		self.fin_file_name = "data/predictions_v" + raw_input("experiment version number") + ".csv"
		#self.pre_process_data()
		self.svm()
		self.logit()

	def svm(self):
		pass

	def logit(self):
		logit = LogisticRegression()
		logit.fit(self.search_inputs.X_train,self.search_inputs.y_train)
		self.logit_preds = logit.predict(self.search_inputs.X_test)

	def relevance_vote(self):
		pass 
	
	def write_file(self):
		# for a singleton model, we just make it a dataframe and write it
		self.fin_df['predictions'] = np.array(self.logit_preds)
		fin_file = fin_df.to_csv(self.fin_file_name,index_label='id')

if __name__ == "__main__":
	predictions = LearnedPrediction() 