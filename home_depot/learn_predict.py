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
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.learning_curve import learning_curve

class SearchInput:
	def __init__(self):
		self.get_params()
		self.read_file()
		self.initial_data_drop()
		self.split_train_target()

	def get_params(self):
		pass

	def read_file(self):
		pass

	def initial_data_drop(self):
		pass

	def split_train_target(self):

class LearnedPrediction():
	def __init__(self):
		self.search_inputs = SearchInput()
		self.pre_process_data()
		self.svm()
		self.logit()

	def pre_process_data(self):
		# make true train and CV split
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.search_inputs.X_train, self.search_inputs.y_train, test_size=0.33, random_state=42)

	def svm(self):
		pass

	def logit(self):

	def relevance_vote(self):
		pass 
	def write_file(self):
		pass

if __name__ == "__main__":
	predictions = LearnedPrediction() 