import numpy as np
import pandas as pd
from glob import glob
import sys 
from collections import defaultdict
import csv
import math

# this file
# 1. loads the data from a user-prompted file
# 2. checks each column type to determine if it's an integer or string
#	- maybe it's safer to simply present the column entry (or, say, 10) and then allow user to determine what type
#	- because some values may evaluate successfully to an integer but may not be in actuality (zipcode)
# 3. with strings, it creates a dictionary and then provides an enumerated value (instead of string)
# 	- EXCEPTION is the false / true. just going to convert to 0 / 1 
# 	- have to keep track of which are ennumerated dictionaries becaues i DO NOT WANT TO SCALE THEM...RIGHT??? (like zipcodes)
#	- will also write the dictionaries to a file so that i can a) reuse them later (:: y-vals) and b) refrence / review them
# 4. with integers, it will [eventually] scale them
# 5. looks for missing values and fixes them...
# 	- fix it options: 
# 		a. figure out the most common and use that
#		b. use some value outside the norm / range (so that it's a new value)
#		c. just use the "mean"
# 6. writes a new file (for reuse / examination)


def read_file(file_name):
    # Read data
    print "reading file data"
    data = pd.read_csv(file_name)    
    return data

def transform_bools(col):
	print "boolean detected. Converting..."
	for i in range(0,len(col)):
		if col[i] == ("TRUE" or "True" or "true"):
			col[i] = 1
		else:
			col[i] = 0
	return col

def write_dict(col):
	print "non boolean string detected. Writing Dictionary"
	print "Converting values from string to Dict enum values"
	unique_vals = col.unique()
	data_dict = {}
	i = 0
	for val in unique_vals:
		if math.isnan(val):
			pass
		else:
			data_dict[val] = i
			i +=1
	print data_dict
	print unique_vals
	col.applymap(data_dict)
	
	return col

def eval_type(col,i, max_i): # will determine if the column of data is an int or a charcter (that needs transformation)
	if i == 0: # is the id column
		return col 
	elif i == max_i: # is the target
		return col
	else:
		try: # basic attempt at "converting" an int to an int. if it's already an int, then no worries; if not, fails and i know it's a string
			col_clean = np.float(col) # use float because a lot of values are already floats
			if clean_col:
				print col_clean
				col_scaled = scale_ints(col_clean)
				return col_scaled

		except ValueError: # then it's a string
			if col[0] == ("TRUE" or "FALSE"): # it's a boolean 
				col_clean = transform_bools(col)
				return col_clean
			else: # then it's a string and needs a dictionary
				col_clean = write_dict(col)
				return col_clean
def write_file(data_clean, fin_file, col_names):
	# write file
	print "writing final file..."
	final_file = data_clean.to_csv(fin_file,index_label='id',float_format='%.3f')

	return final_file

def initialize():
	print "this file will transform raw, poorly formatted data into a train/test/predict-ready file"
	print "----   ----"
	print "Provide file to transform:"
	transform_file = raw_input("< ")	
	#print "TRANSFORM TYPE: (1) == Train, (2) == Test"
	#trans_type = int(raw_input("< "))
	print "----   ----"
	print "Name FINAL FILE:"
	fin_file = raw_input("< ")

	data = read_file(transform_file)
	
	# some annoyingly necessary initialization steps
	data_clean = np.empty((data.shape)) # empty np array
	# going to get the uniqueness factor of each series / column
	# storing in an array that will then be used to determine whether to keep or drop the column
	col_names = list(data.columns.values) # get the column names for file write
	
	# some columns that are numbers but were being treated strangely
	fix_ar = ['VAR_0008', 'VAR_0009', 'VAR_0010', 'VAR_0011', 'VAR_0043','VAR_0230']

	dates = ['VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_O159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0217']
	
	drop_cols = ['VAR_0044', 'VAR_0201', 'VAR_0214']
	print "reading column types and making appropriate conversions"
	

	print "shape before dropping non-unique:"
	print data.shape
	non_unique_cols = []
	for col in col_names:
		print col
		"""
		if col in fix_ar:
			print "called as fix array"
			data_clean[col] = data[col].convert_objects(convert_numeric=True) # KEY LINE -- FORCES conversion, even when there are non-nums like NaN or a fuxed-up string
			print data_clean[col].dtype
			raw_input("pause")
			# convert all NaNs to 0s
			data[col] = np.nan_to_num(data[col]) # just ot make sure 
		"""
		if col in dates:
			try:
				print 'called as in dates'
				data[col].convert_objects(convert_dates='coerce')
				print data[col]
			except ValueError:
				print "date conversion failed"
				drop_cols.append(col)

		# get unique counts
		# if only one unique in column, drop it. 
		unique_count = data[col].unique()	
		print col 
		print unique_count
		if len(unique_count) == 1:
			if col == 'VAR_0526' or col == 'VAR_0529':
				pass
			else:
				drop_cols.append(col)
				non_unique_cols.append(col)
	# drop shit
	data_clean = data.drop(drop_cols, axis = 1)			
	print "shape AFTER dropping non-unique:"
	print data_clean.shape
	print "non-unique cols"
	print non_unique_cols		
	print "all dropped cols: "
	print drop_cols
	
	# now to write dictionary
	col_names = list(data.columns.values) # get the column names for file write

	raw_input('PRESS ENTER TO CONTINUE')
	"""for col in col_names:
		if data[col].dtype is not ('int' or 'float'):
			data[col] = write_dict(data[col])
	"""

	#data_clean = data.select_dtypes(include=['int', 'float'])
	

	# if TRAINING 
	"""
	if trans_type == 1:
		# going to drop all rows with NaN count > 30% of row
		print data_clean.shape
		drop_thresh = int(data_clean.shape[1] * .05)
		print drop_thresh
		data_clean = data_clean.dropna(axis = 0, thresh= drop_thresh)
		print data_clean.shape
		
		raw_input("PRESS ENTER")
	else: # TEST FILE. DO NOTHING (handled below)
		pass
	"""

	

	"""
	total_rows = data_clean.shape[0]
	for val in unique_vals:
		if (val[1][0] / total_rows)  >= .999:  # will need to experiment more with htis
			data_clean.drop([col_names[val]], axis = 1)
		elif (val[1][1] / total_rows)  >= .999:  # will need to experiment more with htis
			data_clean.drop([col_names[val]], axis = 1))
	"""

	final_file = write_file(data_clean, fin_file, col_names) 
	if final_file:
		print "Final File %r written successfully" % final_file
	else:
		print "something broke in file writing"
	

initialize()

