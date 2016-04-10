""
# data_vis.py
# intent:  help visualize distribution of specific data, such as "completeness" of input values
""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualization:
	def __init__(self):
		self.get_params()
		self.read_files()
		self.merge_fields()
		self.calc_basic_info()

	def get_params(self):
		self.root_dir = "data/"
		self.train_fn = "train.csv"
		self.test_fn = "test.csv"
		self.attributes_fn = "attributes.csv"
		self.prod_desc_fn = "product_descriptions.csv"

	def read_files(self):
		self.train_df = pd.read_csv(self.root_dir+self.train_fn, encoding ='ISO-8859-1')
		self.test_df = pd.read_csv(self.root_dir+self.test_fn, encoding ='ISO-8859-1')
		self.prod_desc_df = pd.read_csv(self.root_dir+self.prod_desc_fn, encoding ='ISO-8859-1')
		self.attr_df = pd.read_csv(self.root_dir+self.attributes_fn, encoding ='ISO-8859-1')

	def merge_fields(self):
		# going to keep attributes and prod desc sep for now
		# want to see the distribution of complete attributes before merging them all together
		# TRAIN
		self.train_df_m1 = self.train_df.merge(self.prod_desc_df, how='inner', on='product_uid')
		self.train_df_m2 = self.train_df.merge(self.attr_df, how='inner', on='product_uid')
		# verify merges
		#print self.train_df_m1.columns.values
		#print self.train_df_m2.columns.values
		# TEST 
		self.test_df_m1 = self.test_df.merge(self.prod_desc_df, how='inner', on='product_uid')
		self.test_df_m2 = self.test_df.merge(self.attr_df, how='inner', on='product_uid')

	def calc_basic_info(self):
		# train
		print "TRAIN STUFF"
		print "Overall completeness of attributes:  %f"  % ( float(self.train_df_m2['value'].count()) / float(self.train_df_m2.shape[0]) )

		# split train by "level" of relevance
		# r1 = relevance == 1; r2 == 2, etc
		self.train_df_m2_r1 = self.train_df_m2[self.train_df_m2['relevance'] == 1]

		self.train_df_m2_r2 = self.train_df_m2[self.train_df_m2['relevance'] == 2]
		self.train_df_m2_r3 = self.train_df_m2[self.train_df_m2['relevance'] == 3]
		# OUTPUT completeness
		print "Relevance == 1 completeness of attributes:  %f"  % ( float(self.train_df_m2_r1['value'].count()) / float(self.train_df_m2_r1.shape[0]) )
		print "Relevance == 2 completeness of attributes:  %f"  % ( float(self.train_df_m2_r2['value'].count()) / float(self.train_df_m2_r2.shape[0]) )
		print "Relevance == 3 completeness of attributes:  %f"  % ( float(self.train_df_m2_r3['value'].count()) / float(self.train_df_m2_r3.shape[0]) )
		#self.train_complete_plot = plt.boxplot(data)
		# test 
		print "TRAIN STUFF"
		print "Overall completeness of attributes:  %f"  % ( float(self.test_df_m2['value'].count()) / float(self.test_df_m2.shape[0]) )


	def create_charts(self):
		#plt.show()
		return

if __name__ == "__main__":
	visualization = Visualization() 