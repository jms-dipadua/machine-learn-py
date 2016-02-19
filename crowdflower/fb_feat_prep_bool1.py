from operator import itemgetter
from collections import defaultdict  
from BeautifulSoup import BeautifulSoup

import numpy as np
import pandas as pd
import csv
import sys
import random

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

	data = get_data(file_name)
	print "sample of file data"
	for i in range (0, 5):
		print data[i]
	
	#query_stats = calc_stats(data)

	# file_name = write_query_summary_csv(query_stats, s_file_name)

def start_phase_two():
	
	print "what is the original file name? example: data/train.csv"
	mega_file = raw_input("")
	
	print "what is the summary file name? example: data/sample_bidder_summary.csv"
	file_name = raw_input("")

	print "Name the consolidated_file"
	consolidated_file = raw_input("")
	
	summary_file_name = write_consolidated_csv(file_name, mega_file, consolidated_file)
	if summary_file_name:
		print "ON FIRE!!!"
	else:
		print "facepalm"


def get_data(file_name):
	print "opening csv file"
	with open(file_name, 'r') as csv_file:
		print "sorting raw data into auction lists"
		reader = csv.reader(csv_file)
	return reader



# bidder stats (from dict)
def calc_stats(data):
	print "calculating bidder stats"
	
	"""
		BOOLEANS::
	"""
	
	return query_stats

def write_query_summary_csv(query_stats, s_file_name):
	
	print "writing summary stats csv"

	with open(s_file_name, 'w+') as new_file:
		writer = csv.writer(new_file, lineterminator='\n')
		
		for stat in query_stats:
			all = []
			query_stat = query_stats.get(stat)
			all.append(stat)
			#print bidder
			print query_stat

			for stats in bidder_stats:
				stat = bidder_stats.get(stats)
				all.append(stat)
			#print all
			writer.writerows([all])
	return new_file

def write_consolidated_csv(file_name, mega_file, consolidated_file):
	with open(file_name, 'r') as summary_stats:
		summary_data = pd.read_csv(summary_stats)
		
		with open(mega_file, 'r') as original:
			existing_d = pd.read_csv(original)
			bidder_ids = existing_d['bidder_id']
			
			with open(consolidated_file, 'w+') as new_file:
				writer = csv.writer(new_file, lineterminator='\n')
				
				for bidder in bidder_ids:
					all_bidder_data = []
					match = False
					for i in range(0, len(tracked_bidders)):
						if bidder == tracked_bidders[i]:
							match = True
							matched_i = i
					if match == True:
						all_bidder_data.append(bidder)
						
						#w/ sim bids
						other_fields = total_bids[matched_i], t_bids1[matched_i], t_bids10[matched_i], t_bids100[matched_i], t_bids1000[matched_i], t_bids3000[matched_i], total_auctions[matched_i], total_auc0[matched_i], total_auc1[matched_i], total_auc10[matched_i], total_auc100[matched_i], total_auc1000[matched_i], total_devices[matched_i], avg_device[matched_i], avg_device2[matched_i], total_ips[matched_i], avg_ip[matched_i], avg_ip2[matched_i], total_countries[matched_i], avg_country[matched_i], avg_country2[matched_i], total_urls[matched_i], avg_url[matched_i], avg_url2[matched_i], avg_num_dev[matched_i], avg_num_countries[matched_i], avg_num_ips[matched_i], avg_num_urls[matched_i], simultaneous_bids[matched_i], simultaneous_country[matched_i], simultaneous_country_auc[matched_i]
						for field in other_fields:
							all_bidder_data.append(field)
						writer.writerows([all_bidder_data])
					else:
						all_bidder_data.append(bidder)
						# FOR 11 variables
						other_fields = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
						for field in other_fields:
							all_bidder_data.append(field)
						writer.writerows([all_bidder_data])
	return consolidated_file


######  KICK IT OFF!!! 
#create_sample_file()
initialize()
#start_phase_two()