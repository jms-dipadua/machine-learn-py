from operator import itemgetter
from collections import defaultdict  

import numpy as np
import pandas as pd
import csv
import sys


def initialize(): 
	print "Let's get started"
	print "this will allow you to input a file when ready...\n getting data now..."
	# SAMPLE
	file_name = 'data/sample_bids.csv'
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
		last_bidder = None 
		last_bid_time = None 
		for i in range (bids_read, bids_read + bid_count):
			auction_items.append(auction_sorted[i])
			auction_times.append(auction_sorted[i][5])
			j +=1
			# marks the last bidder (by getting the bidder_id)
			# going to adjust bid_count (for last bidder as a feature)
			# adding in last bid time to calc "time from auction end" (clearly as an estimate)
			# entities that bid in the last few seconds seem like good candidates for snippers 
			# because human would probably see they'd won and not see the snipped bid change...

			if j == bid_count and bid_count > 5:
				last_bid_time = max(auction_times)
				last_bid_index = auction_times.index(max(auction_times))
				last_bidder = auction_sorted[last_bid_index][1]
				#print "last bidder: %r" % last_bidder
				#print "last bid_time: %r" % last_bid_time
				#print "full list of bid times: %r" % auction_times				
			elif j == bid_count and not last_bidder and not last_bid_time:
				# using random symbol to avoid false matches
				last_bidder = ">>>"
				last_bid_time = max(auction_times)
			else:
				pass

		bids_read = bids_read + bid_count
		#print auction_items
		bidder_dict = write_bidder_dict(auction_items, bidder_dict, last_bidder, last_bid_time)

	#print bidder_dict
	#print "total bids_read = %f" % bids_read
	#print "total bidders in all %f auctions is %f" % (bids_read, len(bidder_dict))

	bidder_summary_stats = calc_bidder_stats(bidder_dict)
	#print bidder_summary_stats

	file_name = write_bidder_summary_csv(bidder_summary_stats)

	#print file_name
	#start_phase_two()

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

def write_bidder_dict(auction_items,bidder_dict, last_bidder, last_bid_time):
	print "writing bidder dictionary"
	print last_bidder

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
		

		# normalize a time difference (from last bid)
		try:
			time_diff = abs(int(time_) - int(last_bid_time)) * np.exp(1e-6)
		except ValueError:
			#just to avoid load fail due to empty value:
			time_diff = 30
			print "not an int"
			pass
		
		

			
		if not bidder:
			#print "NEW BIDDER!!"
			device = {device_: 1 }
			bid_time = { 0 : time_ }
			country = {country_: 1}
			ip ={ip_: 1}
			url ={url_: 1}

			# last bidder_controls
			if bidder_ref == last_bidder:
				last_bidder_count = 1
			else:
				last_bidder_count = 0

			# bid time from last bid (as snipe estimate)	
			if (time_diff <= 3):
				snipe_bid_count = 1
			else: 
				snipe_bid_count = 0

			auction_data = {
				'bids': 1, 
				'devices': device, 
				'times': bid_time, 
				'countries': country,
				'ips': ip,
				'urls': url,
				'last_bidder_count': last_bidder_count,
				'snipe_bid_count': snipe_bid_count
			}

			bidder_dict[bidder_ref] = { auction_ref: auction_data }
		else: 
			auction_data = bidder.get(auction_ref)
			# if this check fails, it means we're onto a new auction
			# so we'll need to setup the new auction dict within the bidder's dict
			if not auction_data:
				#print "this is getting called. needs to be fixed."
				#bids = {'bids':  1}
				device = {device_: 1 }
				bid_time = { 0 : time_ }
				country = {country_: 1}
				ip = {ip_: 1}
				url ={url_: 1}

				# last bidder_controls
				if bidder_ref == last_bidder:
					last_bidder_count = 1
				else:
					last_bidder_count = 0

				# bid time from last bid (as snipe estimate)	
				if (time_diff <= 3):
					snipe_bid_count = 1
				else: 
					snipe_bid_count = 0
					
				auction_data = {
					'bids': 1, 
					'devices': device, 
					'times': bid_time, 
					'countries': country,
					'ips': ip,
					'urls': url,
					'last_bidder_count': last_bidder_count,
					'snipe_bid_count': snipe_bid_count
				}

				bidder_dict[bidder_ref][auction_ref] = auction_data 

			else: #if it passes, it means we need to update our data
				
				new_auction_data = update_auction_data(auction_data, bid, last_bidder, bidder_ref, time_diff)
				
				bidder_dict[bidder_ref][auction_ref]= new_auction_data 
					
	return bidder_dict


def update_auction_data(auction_data, bid_data, last_bidder, bidder_ref, time_diff): 
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

	# device check
	if not device_ in auction_data['devices']:
		auction_data['devices'][device_] = 1
	else:
		auction_data['devices'][device_] += 1	
		pass
	
	# country check
	if not country_ in auction_data['countries']:
		auction_data['countries'][country_] = 1
	else:
		auction_data['countries'][country_] += 1
		pass
	
	# IP check
	if not ip_ in auction_data['ips']:
		auction_data['ips'][ip_] = 1
	else:
		auction_data['ips'][ip_] += 1
		pass

	# URL check
	if not url_ in auction_data['urls']:
		auction_data['urls'][url_] = 1
	else:
		auction_data['urls'][url_] += 1
		pass
	
	#last bidder count check
	if bidder_ref == last_bidder:
		auction_data['last_bidder_count'] += 1	

	# bid time from last bid (as snipe estimate)	
	if (time_diff <= 3):
		#auction_data['snipe_bid_count'] += 1
		# testing boolean setting only (equiv to a "sniper" vs count of snipe_attempts)
		auction_data['snipe_bid_count'] += 1
	print auction_data
	exit()
	return auction_data

# bidder stats (from dict)
def calc_bidder_stats(bidder_dict):
	print "calculating bidder stats"
	"""
		DONE * total bids
		DONE  * total auctions (participated in)
		* avg time between bids (by auction?):::  PUNT
		DONE  * total devices used
		DONE  * avg num devices per auction (?)
		DONE  * avg num country per auction
		DONE  * avg num IPs used per auction
		DONE  * total IPs used
		DONE  * total simultaneous_bids
		DONE  * snipe attemps (@ 16s)
		DONE  * total urls used
	"""
	bidder_stats = {}
	bidder_count = 0
	for bidder in bidder_dict:
		bidder_count += 1
		print "NEW BIDDER STARTED ---Bidder: %r---" % bidder_count
		bidder_data = bidder_dict.get(bidder)
		#print bidder_data		
		total_bids = 0
		total_auctions = len(bidder_data)
		total_devices = 0
		total_ips = 0
		total_countries = 0
		sim_bid = 0
		total_last_bids = 0
		total_snipe_bids = 0
		total_urls = 0
		
		
		# we'll define a "simultaneous" bid here
		# == bid times within 5 seconds of one another (across auctions, ignoring-same-auction proximity) 
		auction_count = 0
		for auction in bidder_data:
			auction_count += 1
			print "NEW Auction STARTED ---Auction: %r---" % auction_count
			auction_data = bidder_data.get(auction)
			total_bids += auction_data['bids']
			total_devices += len(auction_data['devices'])
			
			## going to see about getting the exact number of countries and ips, etc 
			## see if that makes a difference. 
			## pending results, will then move to getting a "country" switch boolean
			## based on results
			for country in auction_data['countries']:
				total_countries += auction_data['countries'][country]
			for ip in auction_data['ips']:
				total_ips += auction_data['ips'][ip]
			
			
			total_last_bids += auction_data['last_bidder_count']
			#total_snipe_bids += auction_data['snipe_bid_count']
			# TEST: boolean (redundant but so what)
			
			total_snipe_bids += auction_data['snipe_bid_count']
			for url in auction_data['urls']:
				total_urls += auction_data['urls'][url]
			


			# all of the following for 'simultaneous bid' 
			# but note: this is not the 'counting version' 
			# boolean: simultaneous bidder or not
			
			# test: expanding the sim bid by a little bit
			# going to do a count on sim bids per auction (but not more than one per auction)
			# should reduce the run time a little. 
			"""
			local_sim_bid_count = 0
			for bid_time in auction_data['times']:
				current_eval_t = auction_data['times'].get(bid_time)
				
				if local_sim_bid_count == 1:
						break
				else:
					pass
				
				for tmp_auction in bidder_data:
					tmp_a_data = bidder_data.get(tmp_auction)

					for tmp_time_ref in tmp_a_data['times']:
						tmp_time = tmp_a_data['times'].get(tmp_time_ref)
						
						try: 
							time_diff = abs(int(tmp_time) - int(current_eval_t)) * np.exp(1e-6)
							if (time_diff <= 5) and not (tmp_auction == auction):
								local_sim_bid_count = 1
								sim_bid = sim_bid + local_sim_bid_count
								
						except ValueError:
							print "not an int"
							pass
				"""

		avg_num_dev = total_devices / total_auctions
		avg_num_ips = total_ips / total_auctions
		avg_num_countries = total_countries / total_auctions
		#sim_bid = sim_bid / 2

		# another option is to run a "average snipe" like the avg # of devices, etc
		
		bidder_stats[bidder] = {
			'total_bids': total_bids,
			'total_auctions': total_auctions,
			'total_devices': total_devices,
			'total_ips': total_ips,
			'total_countries': total_countries,
			'avg_num_dev': avg_num_dev,
			'avg_num_countries': avg_num_countries,
			'avg_num_ips': avg_num_ips,
			#'simultaneous_bids': sim_bid,
			'num_last_bids': total_last_bids,
			'num_snipe_bids': total_snipe_bids,
			'total_urls': total_urls
		}
	return bidder_stats

def write_bidder_summary_csv(bidder_summary_stats):
	print "name bidder summary file"
	file_name = raw_input("")
	print "writing bidder summary stats csv"

	#file_name = 'data/bidder_summary.csv'
	with open(file_name, 'w+') as new_file:
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
	return file_name

def write_consolidated_csv(file_name, mega_file, consolidated_file):
	with open(file_name, 'r') as summary_stats:
		summary_data = pd.read_csv(summary_stats)
		tracked_bidders = summary_data['bidder_id']
		total_bids = summary_data['total_bids']
		total_auctions = summary_data['total_auctions']
		total_devices = summary_data['total_devices']
		total_ips = summary_data['total_ips']
		total_countries = summary_data['total_countries']
		avg_num_dev = summary_data['avg_num_dev']
		avg_num_countries = summary_data['avg_num_countries']
		avg_num_ips = summary_data['avg_num_ips']
		#simultaneous_bids = summary_data['simultaneous_bids']
		last_bids = summary_data['num_last_bids']
		num_snipe_bids = summary_data['num_snipe_bids']
		total_urls = summary_data['total_urls']

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
						#w/out sim bids: (+ urls)
						other_fields = total_bids[matched_i], total_auctions[matched_i], total_devices[matched_i], total_ips[matched_i], total_countries[matched_i], avg_num_dev[matched_i], avg_num_countries[matched_i], avg_num_ips[matched_i], last_bids[matched_i], num_snipe_bids[matched_i], total_urls[matched_i]
						#w/ sim bids (+ urls)
						#other_fields = total_bids[matched_i], total_auctions[matched_i], total_devices[matched_i], total_ips[matched_i], total_countries[matched_i], avg_num_dev[matched_i], avg_num_countries[matched_i], avg_num_ips[matched_i], simultaneous_bids[matched_i], last_bids[matched_i], num_snipe_bids[matched_i], total_urls[matched_i]
						for field in other_fields:
							all_bidder_data.append(field)
						writer.writerows([all_bidder_data])
					else:
						all_bidder_data.append(bidder)
						# w/ sim bids + total urls
						# other_fields = [0,0,0,0,0,0,0,0,0,0,0,0]
						# w/out sim bids + total urls
						other_fields = [0,0,0,0,0,0,0,0,0,0,0]
						for field in other_fields:
							all_bidder_data.append(field)
						writer.writerows([all_bidder_data])
	return consolidated_file


######  KICK IT OFF!!! 
initialize()
#start_phase_two()