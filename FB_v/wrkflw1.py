import sys
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier as knn

# POA - 1 
# divide data into two sets of 10 grids
# ex: grid_x will be x <1 and y >0 < 10
# train on all 10 of those grids, get predictions for those grids
# then do the same on a set of y_based grids (where the "clean split" is on y<1 and x> 0 < 10)
# after a set of predictions (KNN) are completed
# will ensemble via a voting 


class City: 
	def __init__(self):
		self.read_file()
		self.gen_train_test()

	def read_file(self):
		print "reading file"
		self.raw_data = pd.read_csv("data/train.csv")

	
