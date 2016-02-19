from operator import itemgetter
from collections import defaultdict  
from BeautifulSoup import BeautifulSoup
# TFIDF & COSINE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# experimental stuff -- word vectors
from gensim import corpora, models, similarities, matutils
from collections import defaultdict

import re
import numpy as np
import pandas as pd
import csv
import sys
import random
import requests
import inflect 


def create_sample_file():
	print "What is the original file name?"
	o_file = raw_input("")
	print "How large of a sample do you want? (Whole Integers, no commas please)"
	random_size = int(raw_input(""))
	print "Name for new sample file?"
	s_file = raw_input("")
	print "opening csv file"
	with open(o_file, 'r') as csv_file:
		#reader = csv.reader(csv_file)
		with open (s_file, 'w+') as sample_file:
			writer = csv.writer(sample_file, lineterminator='\n')
			print "creating random sample set"
			# random sample without replacement
			rand_sample = random.sample(csv_file.readlines(), random_size)		
			print "begining: write file"
			all = []
			for line in rand_sample:
				entries = line.split(',')
				all.append(entries)
			writer.writerows(all)

def initialize(): 
	print "Let's get started"
	print "what is the name of the file to summarize?"
	file_name = raw_input("")
	print "name to-be completed summary file"
	s_file_name = raw_input("")
	print "FILE FOR similarities AND cosine"
	sim_cos_file = raw_input("")

	print "train or test file? "
	sum_type = int(raw_input("1 = train, 2 = test "))

<<<<<<< HEAD
	# query to product title similiarity
	q_pt_sim = calc_similarities(sim_cos_file, 'product_title', 'query')
	# query_full_path to pt_full_path sim
	path_sim = calc_similarities(sim_cos_file, 'q_full_path', 'pt_full_path')

	# query to pt_actual_cat sim
	query_pt_cat_sim = calc_similarities(sim_cos_file, 'query', 'pt_actual_cat')
	# query to full pt_path sim
	query_full_path_sim = calc_similarities(sim_cos_file, 'query', 'pt_full_path')
=======

	query_similarities = calc_similarities(sim_cos_file)
>>>>>>> parent of 4da3bb3... starting to do v13, path sim and more. going to generalize similarity function
	#tfidf_cosine = calc_tfidf_cosine(sim_cos_file)

	query_data = get_data(file_name)
	"""
	print "sample of file data"
	for i in range(0,5):
		print query_data[i]
		terms = query_data[i][1].split(' ')
		print terms
		print "of length: %r" % len(terms)
	"""
	# get some quick insight into data
	"""
	total_1_word = 0
	total_2_word = 0
	total_3_word = 0
	total_4_word = 0
	total_lotsa_word = 0
	for query in query_data:
		terms = query[1].split(' ')

		if len(terms) == 1:
			total_1_word += 1
		elif len(terms) == 2: 
			total_2_word += 1
		elif len(terms) == 3:
			total_3_word += 1
		elif len(terms) == 4:
			total_4_word += 1
		elif len(terms) > 4:
			total_lotsa_word += 1
		else:
			pass
	
	print "1 word queries: %r" % total_1_word
	print "2 word queries: %r" % total_2_word
	print "3 word queries: %r" % total_3_word
	print "4 word queries: %r" % total_4_word
	print "total_lotsa queries: %r" % total_lotsa_word
	"""

	# generate the query-specific stats
	# if this is the train set, then obtain mu & sigma
	# score_mu and score_sigma (from score_variance)
	if sum_type == 1:
		[query_stats, score_mu, score_sigma] = calc_stats(query_data, sum_type)
	# ELSE: sum_type == 2: test
	elif sum_type == 2:
		# get mu and sigma from train run
		score_mu = float(raw_input("Please input mu"))
		score_sigma = float(raw_input("please input sigma"))
		# create mock scores w/ mu & sigma
		mock_score_var = create_mock_score(len(query_data), score_mu, score_sigma)
		# append mock_var_score to query_data
		for i in range(0, len(query_data)): 
			query_data[i].append(float(mock_score_var[i]))
		# continue with the summarization process 
		[query_stats, score_mu, score_sigma] = calc_stats(query_data, sum_type)


	#print query_stats
	# verify results
	print "MU:   ", score_mu
	print "SIGMA:   ", score_sigma


<<<<<<< HEAD
	file_name = write_query_summary_csv(query_stats, q_pt_sim, path_sim, query_pt_cat_sim, query_full_path_sim, s_file_name)
=======
	file_name = write_query_summary_csv(query_stats, query_similarities, s_file_name)
>>>>>>> parent of 4da3bb3... starting to do v13, path sim and more. going to generalize similarity function
	if file_name:
		print "file saved as %r" % file_name

def get_data(file_name):
	print "opening csv file"
	with open(file_name, 'r') as csv_file:
		print "opening and reading file"
		reader = csv.reader(csv_file)
		data_sorted = sorted(reader, key=itemgetter(0), reverse=False)
	return data_sorted

def calc_stats(data, sum_type):
	print "calculating query stats"
	
	"""
		BOOLEANS  ::
			Exact match prod_title
			Partial match prod_title

			Exact match prod_desc
			Partial match prod_desc

		RATIOS :: 
			num_single_word_match :: accumilation of single-word matches from query terms
			mean_1_word_match :: 
				ex: 
				if there are 2 terms in the search
				matches 1st word: that is 1 of 2 terms. 50%
				matches 2nd word: that is 1 of 2 terms. 50%
				mean: 50 + 50 / 2 = 50% (mean percent match)
				ex: 3 terms
				matches 1st: 1 of 3: 33%
				matches 2nd: 1 of 3: 33%
				no match 3rd: 66% / 3 = 22% (mean)
			num_double_ :: same as single
			// num_triple --> LATER (test)
			num_multi :: total number of multi-word matches
			ex: 3 terms
				matches 1st & 2nd: 2 of 3: 66%
				no match 2nd & 3rd: 0%
				 mean == 66% / 2 = 33% (mean)
			mean_multi_match :: mean of num_multi as above
			match_to_length :: a single summarized mean of term(s) matched per total q_term length

			create mimic'd worker score variance 

		FUTURE :: 
			distances between matched words --> mean distances... 
			check for similar (?) --> using tf / idf
			create categories (as dicts) w/ ref to tf / idf (mu & sigma ?)
	"""
	# WRITE A DICT OF QUERY --> LIKE BIDDER_STATS[BIDDER]
	# in a for loop through all the queries

	c = 0 # a counter
	query_stats = {} # initialize query_stats
	score_vars = []
	corpus = []
	anmA_count = 0
	anmA_q_ids = []
	anmB_count = 0
	anmB_q_ids = []

	# tfidf stuff
	"""
	for datum in data: 
		query_raw = datum[1]
		prod_title = datum[2]
		prod_desc = datum[3]

		# add to corpous for tfidf stuff
		corpus.append(query_raw)
		corpus.append(prod_title)
		if len(prod_desc) > 0:
			corpus.append(prod_desc)
	"""

	"""
	#scikit learn's TF/IDF
	tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')
	tfidf = tfv.fit_transform(corpus)
	"""
	for datum in data: 
		query_id = int(datum[0])
		query_raw = datum[1]
		query = query_raw.split(' ')
		query_ln = len(query)
		# for loops using len we want max_index
		max_index = query_ln - 1
		prod_title = datum[2]
		prod_desc = datum[3]

		# to make sure we get the right fields for categories:
		if sum_type == 1:
			#rating = int(datum[4])
			score_var = float(datum[5])
			q_partent_cat = datum[6]
			q_act_cat = datum[7]
			# q_full_cat = datum[8] #later...if we want to do fancy stuff...
			pt_partent_cat = datum[9]
			pt_act_cat = datum[10]
			#pt_full_cat = datum[11] #later. per above
		elif sum_type == 2:
			q_partent_cat = datum[4]
			q_act_cat = datum[5]
			# q_full_cat = datum[6] #later...if we want to do fancy stuff...
			pt_partent_cat = datum[7]
			pt_act_cat = datum[8]
			#pt_full_cat = datum[9] #later. per above
			score_var = float(datum[10])
		else:
			print "UH...WTF???"
			exit()
		
		# for calcuating mu & sigma
		score_vars.append(score_var)

		# add to corpous for tfidf stuff
		corpus.append(query_raw)
		corpus.append(prod_title)
		if len(prod_desc) > 0:
			corpus.append(prod_desc)
		# SOME MORE INITIALIZATIONS
		pt_exact = 0
		pt_part = 0
		pd_exact = 0
		pd_part = 0

		#query category booleans
		if (q_partent_cat or pt_partent_cat)  == '0':
			parent_cat_exact = 0
		elif (q_partent_cat == pt_partent_cat) and not (q_partent_cat or pt_partent_cat) == 0:
			parent_cat_exact = 1
		else: 
			parent_cat_exact = -1
			
		if (q_act_cat or pt_act_cat) == '0':
			act_cat_exact = 0
		elif (q_act_cat == pt_act_cat) and not (q_act_cat or pt_act_cat) == 0 :
			act_cat_exact = 1
		else: 
			act_cat_exact = -1

		#(EX. HTML)
		#if query_id == 290: 
		#	print prod_desc
		prod_title = strip_html(prod_title)
		prod_desc = strip_html(prod_desc)

		# REMOVING SINCE IT'S JUST A PASS THROUGH
		
		#if rating == 4 and len(prod_desc):
		#	empty_pds += 1
	
		#if query_id == 290:
		#	print prod_desc
	
		# check prod_title and prod_desc for exact 
		# partial handled within conts and means
		query_raw = re.compile(query_raw, re.IGNORECASE)

		pt_exact = re.search(query_raw, prod_title)
		if pt_exact:
			pt_exact = 1
			pt_part = 1
		else:
			pt_exact = 0

		pd_exact = re.search(query_raw, prod_desc)
		if pd_exact:
			pd_exact = 1
			pd_part = 1
		else:
			pd_exact = 0

		# count / mean SINGLE
		# title
		[num_sw_mtch_t, mean_1wm_t, one_wm_t] = find_maches(query, query_ln, max_index, prod_title, 1)
		if num_sw_mtch_t > 0:
			pt_part = 1
		# description
		[num_sw_mtch_d, mean_1wm_d, one_wm_d] = find_maches(query, query_ln, max_index, prod_desc, 1)
		if num_sw_mtch_d > 0:
			pd_part = 1
		
		# DOUBLE
		# title
		if query_ln < 2:
			num_dw_mtch_t = pt_exact
			mean_2wm_t = pt_exact
			two_wm_t = pt_exact
		else: 
			[num_dw_mtch_t, mean_2wm_t, two_wm_t] = find_maches(query, query_ln, max_index, prod_title, 2)
		# description
		if query_ln < 2:
			num_dw_mtch_d = pd_exact
			mean_2wm_d = pd_exact
			two_wm_d = pd_exact
		else: 
			[num_dw_mtch_d, mean_2wm_d, two_wm_d] = find_maches(query, query_ln, max_index, prod_desc, 2)

		# MULTI
		# title
		if query_ln < 3:
			num_multi_mtch_t = num_dw_mtch_t
			mean_multi_t = mean_2wm_t
			multi_wm_t = mean_2wm_t
		else: 
			[num_multi_mtch_t, mean_multi_t, multi_wm_t] = find_maches(query, query_ln, max_index, prod_title, query_ln)
		# description
		if query_ln < 3:
			num_multi_mtch_d = num_dw_mtch_d
			mean_multi_d = mean_2wm_d
			multi_wm_d = mean_2wm_d
		else: 
			[num_multi_mtch_d, mean_multi_d, multi_wm_d] = find_maches(query, query_ln, max_index, prod_desc, query_ln)


		# start minium coverage
		if query_ln < 2 and (pt_exact == 1 or pd_exact == 1):
			min_coverage_t = 0 # _t == title
			min_coverage_d = 0 # _d == desc.
			min_coverage_a = 0 # _a == all text
			min_pair_dist_t = 0
			min_pair_dist_d = 0
			min_pair_dist_a = 0
		elif query_ln < 2 and not (pt_exact == 1 or pd_exact == 1):
			min_coverage_t = query_ln # _t == title
			min_coverage_d = query_ln # _d == desc.
			min_coverage_a = query_ln # _a == all text
			min_pair_dist_t = query_ln
			min_pair_dist_d = query_ln
			min_pair_dist_a = query_ln
		else:
			[min_coverage_t, min_coverage_d, min_coverage_a, min_pair_dist_t, min_pair_dist_d, min_pair_dist_a] = calc_min_coverage(query, query_ln, max_index, prod_desc, prod_title)

		# anom_typeA: is 3 or 4 but has no match in either pd or pt
		# because there are only 36 anamolies for each A & B, removing this
		"""
		if rating == (4 or 3) and pt_part == 0 and pd_part == 0:
			anom_typeA = 1
			anmA_count += 1
			anmA_q_ids.append(query_id)
		else:
			anom_typeA = 0
			
		# anom_typeB: is 1 or 2 but has pt_exact
		if rating == (1 or 2) and pt_exact == 1:
			anom_typeB = 1
			anmB_count += 1
			anmB_q_ids.append(query_id)
		else: 
			anom_typeB = 0
		"""
		query_stats[query_id] = {
		# BOOLEANS
			'pt_exact': pt_exact,
			'pt_part': pt_part,
			'pd_exact': pd_exact,
			'pd_part': pd_part,
			'parent_cat_exact': parent_cat_exact,
			'act_cat_exact': act_cat_exact,
			# COUNTS AND MEANS --> for titles and desc? or just desc?
			# v1 :: descriptions only --> test titles next
			'num_sw_mtch_t': num_sw_mtch_t,
			'num_sw_mtch_d': num_sw_mtch_d,
			'mean_1wm_t': mean_1wm_t,
			'mean_1wm_d': mean_1wm_d,
			'num_dw_mtch_t': num_dw_mtch_t,
			'num_dw_mtch_d': num_dw_mtch_d,
			'mean_2wm_t': mean_2wm_t,
			'mean_2wm_d': mean_2wm_d,
			'two_wm_t': two_wm_t,
			'two_wm_d': two_wm_d,
			# --> NOTE: lumping 3 and 4+ wrd queries all together (approx 25 percent of examples)
			'num_multi_mtch_t': num_multi_mtch_t,
			'num_multi_mtch_d': num_multi_mtch_d,  
			'mean_multi_t': mean_multi_t,
			'mean_multi_d': mean_multi_d,
			'multi_wm_t': multi_wm_t,
			'multi_wm_d': multi_wm_d,
			'score_var': score_var,
			'min_coverage_t': min_coverage_t,
			'min_coverage_d': min_coverage_d,
			'min_coverage_a': min_coverage_a,
			'min_pair_dist_t': min_pair_dist_t, 
			'min_pair_dist_d': min_pair_dist_d, 
			'min_pair_dist_a': min_pair_dist_a, 
			'query_ln': query_ln#,
			#'anom_typeA': anom_typeA,
			#'anom_typeB': anom_typeB
			#'rating': rating
			}

	# MU & SIGMA 
	print anmA_count
	print anmA_q_ids
	print anmB_count
	print anmB_q_ids
	mu = np.array(score_vars)
	#print mu.shape
	mu = np.round(np.mean(mu), 4)
	print "MU:   ", mu
	raw_input("PRESS ENTER TO CONTINUE")
	score_sigma = []
	for query_stat in query_stats:
		#sigma = pow(query_stat[5] - mu, 2)
		stats = query_stats.get(query_stat)
		#print "STATS:   ", stats
		#raw_input("PRESS ENTER TO CONTINUE")
		score_sigma.append(pow(stats['score_var'] - mu, 2))
	sigma = np.round(np.mean(score_sigma), 4)

	print "SIGMA:   ", sigma
	raw_input("PRESS ENTER TO CONTINUE")

	return query_stats, mu, sigma

def calc_similarities(file_name, field_a, field_b):
	# trying to calculate "similiaries" as a word-vector... (heh...uh)
	print "calculting similiaries for %r and %r" % (field_a, field_b)

	data = pd.read_csv(file_name)

	# FIRST we do query and product title (second we do categories)
	document = list(data.apply(lambda x:'%s' % (x[field_a]),axis=1))

	# check if '/' in content? 
	for i in range(0, len(document)):
		document[i] = re.sub('/', '', document[i])

	stoplist = set('for a of the and to in with an on oz lbs. lbs ft ft. in. ml inch cu. cu ft. ft up cm oz. mm ounce'.split())
	texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in document]
    
	frequency = defaultdict(int)

	for text in texts:
		for token in text:
			frequency[token] += 1

	texts = [[token for token in text if frequency[token] > 1]
			for text in texts]

	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

	tfidf = models.TfidfModel(corpus)

	tfidf_vals = tfidf[corpus]

	lsi = models.LsiModel(tfidf_vals, id2word=dictionary, num_topics=300)
    
    #indicies
	index = similarities.MatrixSimilarity(lsi[tfidf_vals])

	queries = list(data.apply(lambda x:'%s' % (x[field_b]),axis=1))

	queries_lsi = []
	for query in queries:
		vec_bow = dictionary.doc2bow(query.lower().split())
		vec_lsi = lsi[vec_bow]
		queries_lsi.append(vec_lsi)

	q_sims = []
	for query in queries_lsi:
		query_index = queries_lsi.index(query)
		sims = index[query]
		sim_lis = list(enumerate(sims))
		for i in range(0, len(sim_lis)):
			if sim_lis[i][0] == query_index:
				q_sims.append(sim_lis[i][1])
				break

<<<<<<< HEAD
	return q_sims
=======
<<<<<<< HEAD
	return similarities
=======
	# NEXT WE DO QUERY_CATS AND PT_CATS
	# we'll have two similarity checks: parent and actual
	#	...actually, we'll do similiarty checks in v.next
	#   ... as quick POC, we'll do booleans 

	return query_sim #, cat_parent_sim, cat_act_sim
>>>>>>> parent of 4da3bb3... starting to do v13, path sim and more. going to generalize similarity function
>>>>>>> master
 
def calc_tfidf_cosine(file_name):
	print "calculating cosine similarity"
	data = pd.read_csv(file_name)
	prod_titles = list(data.apply(lambda x:'%s' % (x['product_title']),axis=1))
	queries = list(data.apply(lambda x:'%s' % (x['query']),axis=1))

	# after you konw the rest of this is workign
	# to be supplied in stop_words as a **LIST**
	# 	stoplist = set('for a of the and to in with an on oz lbs. lbs ft ft. in. ml inch cu. cu ft. ft up cm oz. mm ounce'.split())

	# this improved score so using custom stoplist
	stoplist = list('for a of the and to in with an on oz lbs. lbs ft ft. in. ml inch cu. cu ft. ft up cm oz. mm ounce'.split())

	tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'stoplist')
    
    # Fit TFIDF
	tfv.fit(prod_titles)
	prod_title_tfidf =  tfv.transform(prod_titles) 
	
	# transpose for matrix multiplication and division
	pt_tfidf_T = np.transpose(prod_title_tfidf)
	print pt_tfidf_T.shape

	tfv.fit(queries)
	query_tfidf = tfv.transform(queries)
	q_tfidf_T = np.transpose(query_tfidf)
	print q_tfidf_T.shape

	cosine_tfidf = cosine_similarity(pt_tfidf_T[0:1], q_tfidf_T[0:1])
	print cosine_tfidf

	return cosine_tfidf

def strip_html(raw_text):
	text = re.sub('<[^>]*>', '', raw_text)
	text = re.sub('.[^>]*}', '', text)
	return text

def find_maches(query, query_ln, max_index, text, num_words):
	# inflect will convert plurals into singulars
	string_fix = inflect.engine()
	# aggregation count / instantiations
	num_matches = 0 
	mean_matches = 0
	raw_percent_match = 0
	percent_match = 0
	
	#print "current query:  %r" % query

	k = 0
	for i in range(0, query_ln):
		# kill switch
		# I.E. @ max_index, there is no subsquent term, so we can end the loop and return the counts & means
		if (i + num_words) > query_ln:
 				break

		#assigment step: set the query_term for the RE 
		term = query[i]
		if num_words > 1:
			for j in range(1, num_words):
				term = term + ' ' + query[i+j]
				# another kill switch
				if j > query_ln:
					print "ME!!!  MIGHT BE ERROR"
					break
		#print term
		#print text
		# first do a search on the raw word (so pluralizations are okay)
		term_re = re.compile(term, re.IGNORECASE)
		matches = re.findall(term_re, text)
		num_matches += len(matches)
		if matches:
			percent_match += float(num_words) / query_ln
		
		# convert to singular form of noun and do the search again
		converted_term = string_fix.singular_noun(term)
		if converted_term != False:
			term_re = re.compile(converted_term, re.IGNORECASE)
			matches = re.findall(term_re, text)
			num_matches += len(matches)
			if matches:
				percent_match += float(num_words) / query_ln
			
		# add it all up
		mean_matches += percent_match
		raw_percent_match += percent_match

		#print "list of matches: %r" % matches
		#print "number matches: %r    pre-calc'd mean matches: %r" % (num_matches, mean_matches)
		k = i
	mean_matches = mean_matches / (k + 1)
	#print "final mean match percent: %r" % mean_matches
	return num_matches, mean_matches, raw_percent_match
            
def calc_min_coverage(query, query_ln, max_index, prod_desc, prod_title):
	location = []
	all_text = prod_title + prod_desc
	# note that i've already done a check for length 
	#so we can assume that these are all non-0-length distances

	# title 
	for i in range(0, query_ln):
		term = re.compile(query[i])
		temp_loc = re.search(term, prod_title)
		if not temp_loc:
			location.append((i+1) * len(prod_title.split(' '))) # some controlled "large distance"
		else:
			locations = temp_loc.span()
			location.append(locations[0])

	# this is potential fix for miscalcualtion of min_coverage
	location = sorted(location)

	if len(location) > 2:  
		min_dist_t = abs(location[len(location) -2] - location[len(location) -1])

		# calculate min_pair_distance
		min_pair_dist_t = 100000 # arbitrarily large number
		for j in range(0, len(location) - 1):
			for i in range(j+1, len(location) - 1):
				dist_temp = abs(location[j] - location[i])
				if dist_temp < min_pair_dist_t:
					min_pair_dist_t = dist_temp
	else: 
		min_dist_t = abs(location[0] - location[len(location) -1])
		min_pair_dist_t = min_dist_t

	#print "min dist_t:   %r" % min_dist_t

	# reset for description
	del location
	location = []
	# description
	for i in range(0, query_ln):
		term = re.compile(query[i])
		temp_loc = re.search(term, prod_desc)
		if not temp_loc:
			temp_loc = (i+1) * len(prod_desc.split(' ')) # some controlled "large distance"
			location.append(temp_loc)
		else:
			locations = temp_loc.span()
			location.append(locations[0])
	
	if len(location) > 2:
		min_dist_d = abs(location[len(location) -2] - location[len(location) -1])
		# calculate min_pair_distance
		min_pair_dist_d = 100000 # arbitrarily large number
		for j in range(0, len(location) - 1):
			for i in range(j+1, len(location) - 1):
				dist_temp = abs(location[j] - location[i])
				if dist_temp < min_pair_dist_d:
					min_pair_dist_d = dist_temp
	else: 
		min_dist_d = abs(location[0] - location[len(location) -1])
		min_pair_dist_d = min_dist_d
	#print "min dist_d:   %r" % min_dist_d
	# reset for all
	del location
	location = []
	
	# all 
	for i in range(0, query_ln):
		term = re.compile(query[i])
		temp_loc = re.search(term, all_text)
		if not temp_loc:
			temp_loc = (i+1) * len(all_text.split(' ')) # some controlled "large distance"
			location.append(temp_loc)
		else:
			locations = temp_loc.span()
			location.append(locations[0])

	if len(location) > 2:
		min_dist_a = abs(location[len(location) -2] - location[len(location) -1])
		# calculate min_pair_distance
		min_pair_dist_a = 100000 # arbitrarily large number
		for j in range(0, len(location) - 1):
			for i in range(j+1, len(location) - 1):
				dist_temp = abs(location[j] - location[i])
				if dist_temp < min_pair_dist_a:
					min_pair_dist_a = dist_temp
	else: 
		min_dist_a = abs(location[0] - location[len(location) -1])
		min_pair_dist_a = min_dist_a

	#print "min dist_a:  %r" % min_dist_a

	return min_dist_t, min_dist_d, min_dist_a, min_pair_dist_t, min_pair_dist_d, min_pair_dist_a

def create_mock_score(num_queries, mu, sigma):
	mock_scores = []
	for i in range(0, num_queries):
		mock_scores.append(abs(round(random.gauss(mu, sigma), 3)))
	#print mock_scores
	return mock_scores

<<<<<<< HEAD
def write_query_summary_csv(query_stats, q_pt_sim, path_sim, query_pt_cat_sim, query_full_path_sim, s_file_name):	
=======
def write_query_summary_csv(query_stats, query_similarities, s_file_name):	
>>>>>>> parent of 4da3bb3... starting to do v13, path sim and more. going to generalize similarity function
	print "writing summary stats csv"
	#print query_stats # for ordering of fields

	with open(s_file_name, 'w+') as new_file:
		writer = csv.writer(new_file, lineterminator='\n')
		
		i = 0 # for query_similarities
		for stat in query_stats:
			all = []
			query_stat = query_stats.get(stat)
<<<<<<< HEAD
			all.append(stat) 
=======
			all.append(stat)
<<<<<<< HEAD
>>>>>>> master
			all.append(q_pt_sim[i])
			all.append(path_sim[i])
			all.append(query_pt_cat_sim[i])
			all.append(query_full_path_sim[i])
=======
			all.append(query_similarities[i])
>>>>>>> parent of 4da3bb3... starting to do v13, path sim and more. going to generalize similarity function
			#all.append(tfidf_cosine[i])
			#print stat

			for stats in query_stat:
				stat = query_stat.get(stats)
				all.append(stat)
			#print all
			writer.writerows([all])

			i+=1 # for indexing 

	return new_file


######  KICK IT OFF!!! 
#create_sample_file()
initialize()
#start_phase_two()