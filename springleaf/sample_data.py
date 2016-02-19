## THIS FILE HELPS YOU GET A SAMPLE OF DATA
## DOES NOT retain the first line. probably should fix that but fuck it (for now)

import csv
import sys
import random

def create_sample_file():
	print "What is the original file name?"
	o_file = raw_input("")
	print "How large (# of rows) of a sample do you want? (Whole Integers, no commas please)"
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
				# just to see wtf is in this file!
				#print line 
			writer.writerows(all)

create_sample_file()