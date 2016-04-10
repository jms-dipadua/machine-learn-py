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
		get_params()
		read_files()
		merge_fields()
		render_chart()

	def get_params(self):
		self.root_dir = "/data"
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
		# TEST 
		self.test_df_m1 = self.test_df.merge(self.prod_desc_df, how='inner', on='product_uid')
		self.test_df_m2 = self.test_df.merge(self.attr_df, how='inner', on='product_uid')


