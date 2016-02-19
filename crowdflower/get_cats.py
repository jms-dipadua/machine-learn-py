# generally useful/needed
from BeautifulSoup import BeautifulSoup
import csv
import pandas as pd
#requests stuff
import requests
import json 
import time

# walmart API key d7rf2yhcrfm46xq8ep8farvq
	# taxonomy API
	#	http://api.walmartlabs.com/v1/taxonomy?apiKey={d7rf2yhcrfm46xq8ep8farvq}
	# search api
	# http://api.walmartlabs.com/v1/search?apiKey={apiKey}&lsPublisherId={Your LinkShare Publisher Id}&query=ipod

def initialize(): 
	chk_file = raw_input("What is the FILE to get categories for?    ")
	field = raw_input("What FIELD do you want to use?   ")
	s_file = raw_input("What is name of final file?    ")


	file_data = pd.read_csv(chk_file)
	query_ids = list(file_data.apply(lambda x:'%s' % (x['id']),axis=1))

	fields = list(file_data.apply(lambda x:'%s' % (x[field]),axis=1))

	path_data = []
	i = 0 # indexing to query_ids
	num_calls = 0
	num_fails = 0
	for field in fields: 		
		# get the categories from walmart here
		
		r = requests.get('http://api.walmartlabs.com/v1/search?apiKey=32jjrhgapbb4wdwj5ngybc44&lsPublisherId=3253413&query='+str(field)+'&format=json')
		try:
			response = json.loads(r.text)
			response_items = response['items'][0]
			cat_path = response_items.get('categoryPath').split('/')
			parent_cat = cat_path[0]
			actual_cat = cat_path[-1]
			full_path = response_items.get('categoryPath')
			query_path = [parent_cat, actual_cat, full_path]
			#path_data.append(query_path)
		except:
			query_path = [0, 0, 0]
			#path_data.append(query_path)
			num_fails += 1
		path_data.append(query_path)
		num_calls += 1
		print num_calls
		time.sleep(.3)
	
	#print path_data	
	print "total number of calls:   %r" % num_calls
	print "total number of fails:    %r" % num_fails

	saved_file = write_category_file(query_ids, path_data, s_file)

def strip_html(raw_text):
	text = re.sub('<[^>]*>', '', raw_text)
	text = re.sub('.[^>]*}', '', text)
	return text

def write_category_file(query_ids, path_data, s_file):
	with open(s_file, 'w+') as new_file:
		writer = csv.writer(new_file, lineterminator='\n')
		
		i = 0
		for q_id in query_ids:
			all = []
			all.append(q_id)
			for data in path_data[i]:
				all.append(data)
			writer.writerows([all])
			i += 1

	return new_file

initialize()