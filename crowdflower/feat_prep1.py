from operator import itemgetter
from collections import defaultdict  
from BeautifulSoup import BeautifulSoup

import re
import numpy as np
import pandas as pd
import csv
import sys
import random
import requests

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
	print "name to-be-completed summary file"
	s_file_name = raw_input("")

	



def get_data(file_name):
	print "opening csv file"
	with open(file_name, 'r') as csv_file:
		print "opening and reading file"
		reader = csv.reader(csv_file)
		data_sorted = sorted(reader, key=itemgetter(0), reverse=False)
	return data_sorted



            
def create_mock_score(num_queries, mu, sigma):
	mock_scores = []
	for i in range(0, num_queries):
		mock_scores.append(abs(round(random.gauss(mu, sigma), 3)))
	#print mock_scores
	return mock_scores


def write_query_summary_csv(query_stats, s_file_name):
	
	print "writing summary stats csv"

	with open(s_file_name, 'w+') as new_file:
		writer = csv.writer(new_file, lineterminator='\n')
		
		for stat in query_stats:
			all = []
			query_stat = query_stats.get(stat)
			all.append(stat)
			#print stat

			for stats in query_stat:
				stat = query_stat.get(stats)
				all.append(stat)
			#print all
			writer.writerows([all])
	return new_file


######  KICK IT OFF!!! 
#create_sample_file()
initialize()
#start_phase_two()