from operator import itemgetter
from collections import defaultdict  

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
	print "name bidder summary file"
	s_file_name = raw_input("")

	# FULL
	#file_name = 'data/bids.csv'
	auction_sorted = get_data(file_name)
	print "sample of sorted data"
	for i in range (0, 5):
		print auction_sorted[i]

	auction_dict = write_auction_dict(auction_sorted)
	print "there are %s different auctions" % len(auction_dict)
	# loop through auction dictionary
	# get bid count for each auction and read as a chunck
	# thereby tracking start / end of each auction to sort by bidder & get stats, etc.
	bids_read = 0
	bidder_dict = {}
	# dictionaries aren't sorted but we need the sort order
	# so this iterates through each auction in the dic sorted (to match the sort of the sorted data)
	i = 0
	for auction in sorted(auction_dict.keys()):
		bid_count = auction_dict.get(auction)
		#print "auction %r has total %f" % (auction, bid_count)
		auction_items = []
		# isolate the auction items from *this* auction 
		# for sorting and stat collection
		#	going to use the following to mark the last bid (and use as a counter)
		j = 0
		auction_times = []
		
		for i in range (bids_read, bids_read + bid_count):
			auction_items.append(auction_sorted[i])
			auction_times.append(auction_sorted[i][5])
			j +=1
			# marks the last bidder (by getting the bidder_id)
			# going to adjust bid_count (for last bidder as a feature)
			# adding in last bid time to calc "time from auction end" (clearly as an estimate)
			# entities that bid in the last few seconds seem like good candidates for snippers 
			# because human would probably see they'd won and not see the snipped bid change...

		bids_read = bids_read + bid_count
		#print auction_items
		bidder_dict = write_bidder_dict(auction_items, bidder_dict)

	bidder_summary_stats = calc_bidder_stats(bidder_dict)

	file_name = write_bidder_summary_csv(bidder_summary_stats, s_file_name)

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
		auction_sort = sorted(reader, key=itemgetter(2), reverse=False)
	return auction_sort

def write_auction_file(sorted_data, file_name):
	# would use this for long-running processes (way of "getting back" to where i was...TBD)
	#with open(file_name, 'w+'):
	pass

def write_auction_dict(sorted_data):
	print "creating auctions dict"
	auction_dict = {}
	count = 0
	for i in range (0, len(sorted_data)):
		temp_val = sorted_data[i][2]
		auction = auction_dict.get(temp_val)
		# if that doesn't return a value, then it's a new auction		
		if not auction:
			#print "new auction entry"
			auction_dict[temp_val] = 1
			count = count+1
			#print "current total count is %f" % count
		else: 
			#print "existing entry for %s" % temp_val
			auction_dict[temp_val] = auction_dict[temp_val] + 1
			#print "current count for %s is %r" % (temp_val, auction_dict[temp_val])
	return auction_dict

def write_bidder_dict(auction_items,bidder_dict):
	print "writing bidder dictionary"

	# loop through each bidder (with auction list sorted by bidder_ref)
	for bid in sorted(auction_items, key=itemgetter(1)):
		auction_ref = bid[2]
		bidder_ref = bid[1]
		bidder = bidder_dict.get(bidder_ref)
		device_ = bid[4]
		time_ = bid[5]
		country_ = bid[6]
		ip_ = bid[7]
		url_ = bid[8]
	
		if not bidder:
			#print "NEW BIDDER!!"
			device = {0: device_}
			bid_time = {0: time_ }
			country = {0: country_}
			ip = {0: ip_}
			url ={0: url_}

			auction_data = {
				'bids': 1, 
				'devices': device, 
				'times': bid_time, 
				'countries': country,
				'ips': ip,
				'urls': url,
			}

			bidder_dict[bidder_ref] = { auction_ref: auction_data }
		else: 
			auction_data = bidder.get(auction_ref)
			# if this check fails, it means we're onto a new auction
			# so we'll need to setup the new auction dict within the bidder's dict
			if not auction_data:
				device = {0: device_}
				bid_time = {0: time_ }
				country = {0: country_}
				ip = {0: ip_}
				url ={0: url_}

				auction_data = {
					'bids': 1, 
					'devices': device, 
					'times': bid_time, 
					'countries': country,
					'ips': ip,
					'urls': url,
				}

				bidder_dict[bidder_ref][auction_ref] = auction_data 

			else: #if it passes, it means we need to update our data
				new_auction_data = update_auction_data(auction_data, bid)
				bidder_dict[bidder_ref][auction_ref]= new_auction_data 
					
	return bidder_dict


def update_auction_data(auction_data, bid_data): 
	print "updating existing auction data for existing bidder"
	device_ = bid_data[4]
	time_ = bid_data[5]
	country_ = bid_data[6]
	ip_ = bid_data[7]
	url_ = bid_data[8]

	# increase bid count
	auction_data['bids'] += 1
	# add new time
	auction_data['times'][len(auction_data['times'])] = time_ 

	# DIF FROM ORIGINAL
	# RATHER THAN DO A DEVICE CHECK
	# WE JUST SET THE VALUE LIKE WE DO W/ TIME (ABOV)
	# WE'LL JUST USE LIKE INDICES TO MAKE COMPARISIONS

	# device check
	auction_data['devices'][len(auction_data['devices'])] = device_
	
	# country check
	auction_data['countries'][len(auction_data['countries'])] = country_
	
	# IP check
	auction_data['ips'][len(auction_data['devices'])] = ip_
	
	# URL check
	auction_data['urls'][len(auction_data['urls'])] = url_
	
	return auction_data

# bidder stats (from dict)
def calc_bidder_stats(bidder_dict):
	print "calculating bidder stats"
	"""
		CHANGES TO BOOLEANS::
		avg_IP > 1
		avg_country > 1
		avg_device > 1
		total_auc > 0
		total_auc > 10
		total_auc > 100
		total_auc > 1000
		total_bids > 1
		total_bids > 10
		total_bids > 100
		total_bids > 1000
		total_bids > 3000
		avg_urls > 1
		avg_urls > 2
		avg_urls > 5
		sim_bid || 0, 1
		sim_bid_country || 0,1
	"""
	bidder_stats = {}
	bidder_count = 0
	for bidder in bidder_dict:
		bidder_count += 1
		print "NEW BIDDER STARTED ---Bidder: %r---" % bidder_count
		bidder_data = bidder_dict.get(bidder)

		total_bids = 0
		t_bids1 = 0
		t_bids10 = 0
		t_bids100 = 0
		t_bids1000 = 0
		t_bids3000 = 0
		total_auctions = len(bidder_data)
		total_auc0 = 0
		total_auc1 = 0
		total_auc10 = 0
		total_auc100 = 0
		total_auc1000 = 0
		total_devices = 0
		avg_device = 0
		avg_device2 = 0
		total_countries = 0
		avg_country = 0
		avg_country2 = 0
		total_ips = 0
		avg_ip = 0
		avg_ip2 = 0
		total_urls = 0
		avg_url = 0
		avg_url2 = 0
		sim_bid = 0
		sim_bid_country = 0
		sim_bid_country_auc = 0
		
		# we'll define a "simultaneous" bid here
		# == bid times within 5 seconds of one another (across auctions, ignoring-same-auction proximity) 
		auction_count = 0
		for auction in bidder_data:
			auction_count += 1
			# print "NEW Auction STARTED ---Auction: %r---" % auction_count
			auction_data = bidder_data.get(auction)
			total_bids += auction_data['bids']
			#  NOTE - THESE USE THE NEW INDEX APPROACH (V2)
				## FOR BOOLEAN RUN: -- get the unique entries and count them once 
				## BUT: because we don't know the length of time an auction goes, we still have to do the sim_bid / sim_bid_country check separately
			total_devices += len(set(auction_data['devices']))
			total_countries += len(set(auction_data['countries']))
			total_ips += len(set(auction_data['ips']))
			total_urls += len(set(auction_data['urls']))
			
			# all of the following for 'simultaneous bid' 
			# but note: this is not the 'counting version' 
			# boolean: simultaneous bidder or not
			
			# test: see if the bidder is bidding simultaneously
				# AND see if the bidder is doing so from another country
			# going to do a count on sim bids per auction (but not more than one per auction)
			# should reduce the run time a little. 
			# local boolean kill switch
			
			# tracker for index
			current_index = 0
			for bid_time in auction_data['times']:
				current_eval_t = auction_data['times'].get(bid_time)
				current_eval_country = auction_data['countries'][current_index]

				
				# kill switch for loop
					# BOTH
					#if sim_bid_country == 1 and sim_bid == 1:
					# ONLY sim_bid_country_auc 
				if sim_bid_country_auc  == 1:
					break
				# increment current_index now (for code legibility)
				current_index += 1
				
				# here we'll do something similar to what we did above
				# we'll get the index of the bid time 
				# see if it's a sim bid (if so, set sim bid's boolean == 1)
				# we'll also see if it's a different country
				#	(if so, set sim_bid_country == 1)
				# then we'll break from the entire loop (for this bidder)
								
				for tmp_auction in bidder_data:
					tmp_a_data = bidder_data.get(tmp_auction)

					# kill switch for loop
					# BOTH
					#if sim_bid_country == 1 and sim_bid == 1:
					# ONLY sim_bid_country
					#if sim_bid_country == 1:
					#	break
					# ONLY sim_bid_country_auc 
					if sim_bid_country_auc == 1:
						break
					tmp_index = 0
					for tmp_time_ref in tmp_a_data['times']:

						tmp_time = tmp_a_data['times'].get(tmp_time_ref)						
						tmp_country = tmp_a_data['countries'][tmp_index]

						# kill switch for loop
						# BOTH
						#if sim_bid_country == 1 and sim_bid == 1:
						# ONLY sim_bid_country_auc 
						if sim_bid_country_auc  == 1:
							break
						# increment tmp_index (clarity)
						tmp_index += 1
						
						try: 
							time_diff = abs(int(tmp_time) - int(current_eval_t)) * np.exp(1e-6)
							# general simultaneous bid  :: do 3 seconds?
							#if (time_diff <= 5) and not (tmp_auction == auction):
								#sim_bid = 1
							# simultaneous bid but in another country --> sim_bid_country_auc (now)
							if (time_diff <= 5) and not (tmp_country == current_eval_country) and not (tmp_auction == auction):
									sim_bid_country_auc  = 1
						except ValueError:
							print "not an int"
							pass
						
						## END TIME LOOP 						
		avg_num_dev = total_devices / total_auctions
		avg_num_ips = total_ips / total_auctions
		avg_num_countries = total_countries / total_auctions
		avg_num_urls = total_urls / total_auctions
		"""
		CHANGES TO BOOLEANS::
		avg_IP > 1
		avg_IP > 2
		avg_country > 1
		avg_device > 1
		total_auc > 0
		total_auc > 10
		total_auc > 100
		total_auc > 1000
		total_bids > 1
		total_bids > 10
		total_bids > 100
		total_bids > 1000
		total_bids > 3000
		avg_urls > 1
		avg_urls > 2
		avg_urls > 5
		sim_bid || 0, 1
		sim_bid_country || 0,1
		"""
		if avg_num_urls > 1:
			avg_urls = 1
			if avg_num_urls > 2:
				avg_urls2 = 1

		if avg_num_ips > 1:
			avg_ip = 1
			if avg_num_ips > 2:
				avg_ip2 = 1
		
		if avg_num_countries > 1:
			avg_country = 1  
			if avg_num_countries > 2:
				avg_country2 = 1
		
		if avg_num_dev > 1:
			avg_device = 1
			if avg_num_dev > 2:
				avg_device2 = 1
		
		if total_auctions > 0:
			total_auc0 = 1
			if total_auctions > 10:
				total_auc10 = 1
				if total_auctions >= 100:
					total_auc100 = 1
					if total_auctions >= 1000:
						total_auc1000 = 1 
		
		if total_bids > 1:
			t_bids1 = 1
			if total_bids > 10:
				t_bids10 = 1
				if total_bids >= 100:
					t_bids100 = 1
					if total_bids >= 1000:
						t_bids1000 = 1
						if total_bids >= 3000:
							t_bids3000 = 1

		bidder_stats[bidder] = {
			'total_bids': total_bids,
			't_bids1': t_bids1,
			't_bids10': t_bids10,
			't_bids100': t_bids100,
			't_bids1000': t_bids1000,
			't_bids3000': t_bids3000,
			'total_auctions': total_auctions,
			'total_auc0': total_auc0,
			'total_auc1': total_auc1,
			'total_auc10': total_auc10,
			'total_auc100': total_auc100,
			'total_auc1000': total_auc1000,
			'total_devices': total_devices,
			'avg_device': avg_device,
			'avg_device2': avg_device2,
			'total_countries': total_countries,
			'avg_country': avg_country,
			'avg_country2': avg_country2,
			'total_ips': total_ips,
			'avg_ip': avg_ip,
			'avg_ip2': avg_ip2,
			'total_urls': total_urls,
			'avg_url': avg_url,
			'avg_url2': avg_url2,
			'avg_num_dev': avg_num_dev,
			'avg_num_countries': avg_num_countries,
			'avg_num_ips': avg_num_ips,
			'avg_num_urls': avg_num_urls,
			'simultaneous_bids': sim_bid,
			'simultaneous_country': sim_bid_country,
			'sim_bid_country_auc': sim_bid_country_auc
		}
	return bidder_stats

def write_bidder_summary_csv(bidder_summary_stats, s_file_name):
	
	print "writing bidder summary stats csv"

	with open(s_file_name, 'w+') as new_file:
		writer = csv.writer(new_file, lineterminator='\n')
		
		for bidder in bidder_summary_stats:
			all = []
			bidder_stats = bidder_summary_stats.get(bidder)
			all.append(bidder)
			#print bidder
			print bidder_stats

			for stats in bidder_stats:
				stat = bidder_stats.get(stats)
				all.append(stat)
			#print all
			writer.writerows([all])
	return new_file

def write_consolidated_csv(file_name, mega_file, consolidated_file):
	with open(file_name, 'r') as summary_stats:
		summary_data = pd.read_csv(summary_stats)
		tracked_bidders = summary_data['bidder_id']
		total_bids = summary_data['total_bids']
		t_bids1 = summary_data['t_bids1']
		t_bids10 = summary_data['t_bids10']
		t_bids100 = summary_data['t_bids100']
		t_bids1000 = summary_data['t_bids1000']
		t_bids3000 = summary_data['t_bids3000']
		total_auctions = summary_data['total_auctions']
		total_auc0 = summary_data['total_auc0']
		total_auc1 = summary_data['total_auc1']
		total_auc10 = summary_data['total_auc10']
		total_auc100 = summary_data['total_auc100']
		total_auc1000 = summary_data['total_auc1000']
		total_devices = summary_data['total_devices']
		avg_device = summary_data['avg_device']
		avg_device2 = summary_data['avg_device2']
		total_ips = summary_data['total_ips']
		avg_ip = summary_data['avg_ip']
		avg_ip2 = summary_data['avg_ip2']
		total_countries = summary_data['total_countries']
		avg_country = summary_data['avg_country']
		avg_country2 = summary_data['avg_country2']
		total_urls = summary_data['total_urls']
		avg_url = summary_data['avg_url']
		avg_url2 = summary_data['avg_url2']
		avg_num_dev = summary_data['avg_num_dev']
		avg_num_countries = summary_data['avg_num_countries']
		avg_num_ips = summary_data['avg_num_ips']
		avg_num_urls = summary_data['avg_num_urls']
		simultaneous_bids = summary_data['simultaneous_bids']
		simultaneous_country = summary_data['simultaneous_country']
		simultaneous_country_auc = summary_data['sim_bid_country_auc']

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
#initialize()
start_phase_two()