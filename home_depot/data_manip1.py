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
import enchant.checker
from enchant.checker.CmdLineChecker import CmdLineChecker

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
		self.lower_case() # this was a step in stemmer but going to apply as stand-alone
		# self.initial_data_drop() # if applicable (place holder for future)
		#self.spell_check()
		self.gen_stems()
		self.gen_tfidf()
		self.calc_cosine_sim()
		self.calc_kw_ratio()
		self.data_drop() # drop unneeded cols
		self.write_file()

	def set_intial_params(self):
		self.file_dir = "data/"
		self.raw_file = raw_input("train or test:   ") 
		self.raw_file_name = self.raw_file + '.csv'
		self.prod_des_file = "product_descriptions.csv"
		self.version_num = raw_input("Version Number of Transformation: ")
		self.fin_file = self.file_dir + self.raw_file + "_" + self.version_num + ".csv"

	def read_file(self):
		self.dataframe = pd.read_csv(self.file_dir+self.raw_file_name, encoding ='ISO-8859-1') # same as latin-1
		# then read in product description file and merge it into main dataframe
		self.product_description = pd.read_csv(self.file_dir+self.prod_des_file, encoding ='ISO-8859-1') # same as latin-1
		self.dataframe = self.dataframe.merge(self.product_description, how='inner', on='product_uid')	


	def lower_case(self):
		if not self.dataframe['product_title'].empty:
			self.dataframe['product_title'] = self.dataframe['product_title'].str.lower().str.split()
		if not self.dataframe['search_term'].empty:  # this is is only applicable for spellechecked stuff
			self.dataframe['search_term'] = self.dataframe['search_term'].str.lower().str.split()
		if not self.dataframe['product_description'].empty:
			self.dataframe['product_description'] = self.dataframe['product_description'].str.lower().str.split()
		# this is is only applicable for spellechecked stuff
		if not self.dataframe['search_terms_fixed'].empty:  
			self.dataframe['search_terms_fixed'] = self.dataframe['search_terms_fixed'].str.lower().str.split()

	def gen_stems(self):
		# before steming we'll remove stop words
		self.stopwords_main = stopwords.words('english')
		self.stemmer = SnowballStemmer('english').stem
		# mod stopwords a bit  # maybe later remove more measurement related words
		#self.stopwords_alt = list('oz lbs. lbs ft ft. in. ml inch cu. cu ft. ft up cm oz. mm ounce no. of or gal. to'.split())
		# not as surgical as it could be but "easier" 
		# note that we stem here as well as remove stop words
		if not self.dataframe['product_title'].empty:
			#self.dataframe['product_title'] = self.dataframe['product_title'].str.lower().str.split()
			self.dataframe['product_title'].apply(lambda x: [self.stemmer(item) for item in x if item not in self.stopwords_main])
		if not self.dataframe['search_term'].empty:
			#self.dataframe['search_term'] = self.dataframe['search_term'].str.lower().str.split()
			self.dataframe['search_term'].apply(lambda x: [self.stemmer(item) for item in x if item not in self.stopwords_main])
		if not self.dataframe['product_description'].empty:
			self.dataframe['product_description'].apply(lambda x: [self.stemmer(item) for item in x if item not in self.stopwords_main])
		# other stuff worth investigating is splitting on - becauset here are things like 3-piece BUT those could be useful so...yeah... 
		# sanity check
		#print self.dataframe['product_title']
		if not self.dataframe['search_terms_fixed'].empty:  
			self.dataframe['search_terms_fixed'].apply(lambda x: [self.stemmer(item) for item in x if item not in self.stopwords_main])
	
	def gen_tfidf(self):
		self.tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1)#,
            #stop_words = 'stoplist') # can MAYBE go this route w/ the measurement list (above)

		if not self.dataframe['product_title'].empty:
			prod_title = list(self.dataframe.apply(lambda x:'%s' % (x['product_title']), axis=1 ))
			self.prod_title_tfidf = self.tfv.fit_transform(prod_title)
			#self.prod_title_tfidf =  self.tfv.transform(prod_title) 
			
		if not self.dataframe['search_term'].empty:
			prod_query = list(self.dataframe.apply(lambda x:'%s' % (x['search_term']), axis=1 ))
			self.prod_query_tfidf =  self.tfv.transform(prod_query)

		if not self.dataframe['search_terms_fixed'].empty:
			prod_query = list(self.dataframe.apply(lambda x:'%s' % (x['search_terms_fixed']), axis=1 ))
			self.prod_query_fixes_tfidf =  self.tfv.transform(prod_query)

		# now do this all again for product description
		if not self.dataframe['product_description'].empty:
			prod_des = list(self.dataframe.apply(lambda x:'%s' % (x['product_description']), axis=1 ))
			self.prod_des_tfidf = self.tfv.fit_transform(prod_des)

		if not self.dataframe['search_term'].empty:
			prod_query = list(self.dataframe.apply(lambda x:'%s' % (x['search_term']), axis=1 ))
			self.des_query_tfidf =  self.tfv.transform(prod_query)

		if not self.dataframe['search_terms_fixed'].empty:
			prod_query = list(self.dataframe.apply(lambda x:'%s' % (x['search_terms_fixed']), axis=1 ))
			self.des_query_fixes_tfidf =  self.tfv.transform(prod_query)
			
	def calc_cosine_sim(self):
		# product titles to search term
		self.prod_query_raw_cosine_tfidf = np.zeros(self.prod_query_tfidf.shape[0])
		self.prod_query_fixes_cosine_tfidf = np.zeros(self.prod_query_tfidf.shape[0])
		# prod description to search term 
		self.des_query_raw_cosine_tfidf = np.zeros(self.prod_query_tfidf.shape[0])
		self.des_query_fixes_cosine_tfidf = np.zeros(self.prod_query_tfidf.shape[0])
		for i in range(self.prod_query_tfidf.shape[0]):
			self.prod_query_raw_cosine_tfidf[i]=cosine_similarity(self.prod_query_tfidf[i,:], self.prod_title_tfidf[i,:])
			self.prod_query_fixes_cosine_tfidf[i]=cosine_similarity(self.prod_query_fixes_tfidf[i,:], self.prod_title_tfidf[i,:])
			self.des_query_raw_cosine_tfidf[i]=cosine_similarity(self.des_query_tfidf[i,:], self.prod_des_tfidf[i,:])
			self.des_query_fixes_cosine_tfidf[i]=cosine_similarity(self.des_query_fixes_tfidf[i,:], self.prod_des_tfidf[i,:])
		self.dataframe['prod_query_raw_cosine_tfidf'] = self.prod_query_raw_cosine_tfidf
		self.dataframe['prod_query_fixes_cosine_tfidf'] = self.prod_query_fixes_cosine_tfidf
		self.dataframe['des_query_raw_cosine_tfidf'] = self.des_query_raw_cosine_tfidf
		self.dataframe['des_query_fixes_cosine_tfidf'] = self.des_query_fixes_cosine_tfidf

	def calc_kw_matches(self):
		# it will be a match on a) the keyword phrase and then b) the individual words within the search query
		# for v.now, just apply to the fixed search terms... ??? 
		kw_matches = np.zeros(self.dataframe['search_terms_fixed'].shape[0])
		for i in range(self.dataframe['search_terms_fixed'].shape[0]):
			kws_matched = 0
			keywords = self.dataframe['search_terms_fixed'].iloc[i]
			# check for phase in both product title and product description
			if keywords in self.dataframe['product_title'].iloc[i]:
				kws_matched += 1
			if keywords in self.dataframe['product_description'].iloc[i]:
				kws_matched += 1
			# then check for individuals 
			for keyword in keywords:
				if keyword in self.dataframe['product_title'].iloc[i]:
					kws_matched += 1
				if keyword in self.dataframe['product_description'].iloc[i]:
					kws_matched += 1
			# get the ratio (into the array)
			kw_matches[i] = kws_matched
		# after all is said and done, set the dataframe to the ratio
		self.dataframe['kw_matches'] = kw_matches

	
	def data_drop(self):
		self.dataframe = self.dataframe.drop(['product_title', 'search_term', 'search_terms_fixed', 'product_description'], axis=1)

	def write_file(self):
		final_file = self.dataframe.to_csv(self.fin_file,index=False)


if __name__ == "__main__":
	transformer = Transformer() 