""
# data_manip1.py
# intent:  transform raw data into algorithm "ready" data 
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

# TFIDF & COSINE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# for spell checking # PyEnchant
import enchant 

# for stemming and stopwords # PyStemmer + snowballstemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize # not using ATM
from nltk.corpus import stopwords  #  DON'T FORGET TO DOWNLOAD IT! :)  ## nltk.download("stopwords")

# other utility libraries # might not use actually... 
from operator import itemgetter
from collections import defaultdict  
from BeautifulSoup import BeautifulSoup


class Transformer:
	def __init__(self):
		self.set_intial_params()
		self.read_file()
		# self.initial_data_drop() # if applicable (place holder for future)
		#self.spell_check()
		self.gen_stems()
		self.gen_tfidf()
		self.calc_cosine_sim()
		self.write_file()

	def set_intial_params(self):
		self.file_dir = "data/"
		self.raw_file_name = raw_input("Enter File Name (no directory):   ") 
		self.version_num = "v" + raw_input("Version Number of Transformation: ")
		self.fin_file = self.file_dir + self.raw_file_name + "_" + self.version_num

	def read_file(self):
		self.dataframe = pd.read_csv(self.file_dir+self.raw_file_name, encoding ='ISO-8859-1') # same as latin-1

	def spell_check(self):
		# going to work on this part after steming ("just cuz")
		self.spll_chk = enchant.Dict("en_US")

	def gen_stems(self):
		# before steming we'll remove stop words
		self.stopwords_main = stopwords.words('english')
		self.stemmer = SnowballStemmer('english').stem
		# mod stopwords a bit  # maybe later remove more measurement related words
		#self.stopwords_alt = list('oz lbs. lbs ft ft. in. ml inch cu. cu ft. ft up cm oz. mm ounce no. of or gal. to'.split())
		# not as surgical as it could be but "easier" 
		# note that we stem here as well as remove stop words
		if not self.dataframe['product_title'].empty:
			self.dataframe['product_title'] = self.dataframe['product_title'].str.lower().str.split()
			self.dataframe['product_title'].apply(lambda x: [self.stemmer(item) for item in x if item not in self.stopwords_main])
		if not self.dataframe['search_term'].empty:
			self.dataframe['search_term'] = self.dataframe['search_term'].str.lower().str.split()
			self.dataframe['search_term'].apply(lambda x: [self.stemmer(item) for item in x if item not in self.stopwords_main])
		# other stuff worth investigating is splitting on - becauset here are things like 3-piece BUT those could be useful so...yeah... 
		# sanity check
		#print self.dataframe['product_title']
	
	def gen_tfidf(self):
		self.tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1)#,
            #stop_words = 'stoplist') # can MAYBE go this route w/ the measurement list (above)

		if not self.dataframe['product_title'].empty:
			prod_title = list(dataframe['product_title'].apply(lambda x:'%s' % (x),axis=1))
			self.tfv.fit(prod_title)
			self.prod_title_tfidf =  self.tfv.transform(prod_title) 
			# transpose for matrix multiplication and division
			self.prod_title_tfidf = np.transpose(self.prod_title_tfidf)
		
		if not self.dataframe['search_term']:
			prod_query = list(dataframe['search_term'].apply(lambda x:'%s' % (x),axis=1))
			self.tfv.fit(prod_query)
			# transpose (as above)
			self.prod_query_tfidf =  self.tfv.transform(prod_query)
			self.prod_query_tfidf = np.transpose(self.prod_query_tfidf)
	
	def calc_cosine_sim(self):
		#pt_tfidf_T = np.transpose(prod_title_tfidf)
		self.cosine_tfidf = cosine_similarity(prod_title_tfidf[0:1], prod_query_tfidf[0:1])

	def write_file(self):
		final_file = self.dataframe.to_csv(self.fin_file,index_label='id')

		return 

if __name__ == "__main__":
	transformer = Transformer() 