""
# data_manip1.py
# intent:  transform raw data into algorithm "ready" data 
# why: we can isolate the pipeline by separating transformation from actual agorithmic learning
# plus: we won't have to waste computation cycles (and time) by re-doing steming, etc every time we wnat to run our models
# i.e. we can measure twice and cut once (or revisit the raw if need be)
# input: raw data :: train and test.csv
# output:  train_clean and test_clean.csv
""
# general purpose libraries
import sys
import numpy as np
import pandas as pd

# TFIDF & COSINE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# for spell checking # PyEnchant
import enchant 

# for stemming and stopwords # PyStemmer + snowballstemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# other utility libraries # might not use actually... 
from operator import itemgetter
from collections import defaultdict  
from BeautifulSoup import BeautifulSoup


class Transformer:
	def __init__(self):
		self.set_intial_params()
		self.read_file()
		# self.initial_data_drop() # if applicable (place holder for future)
		self.spell_check()
		self.gen_stems()
		self.write_file()

	def set_intial_params(self):
		self.file_dir = "/data/"
		self.raw_file_name = raw_input("Enter File Name (no directory):   ") 
		self.version_num = "v" + raw_input("Version Number of Transformation: ")
		self.fin_file = self.file_dir + self.raw_file_name + "_" + self.version_num

	def read_file(self):
		self.raw_data = pd.read_csv(self.root_dir+self.raw_file_name)

	def spell_check(self):
		self.spll_chk = enchant.Dict("en_US")

	def gen_stems(self):
		self.stopwrds = stopwords.words('english')
		self.stemmer = SnowballStemmer('english').stem

	def write_file(self):


if __name__ == "__main__":
	transformer = Transformer() 