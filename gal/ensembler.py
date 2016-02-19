
"""
ensembler.py
Grasp, Lift, Release @ Kaggle
__author__ : jms. 
Summary: 
 - Ensembles multiple models for one aggregate (Averaged) output prediction file

"""
import numpy as np
import pandas as pd

final_file = raw_input("Name FINAL OUTPUT file:   ")

num_files = int(raw_input("How many files do you want to average?   "))
#num_classes = int(raw_input("How many CLASSIFICATIONS are in your prediction files?   "))

cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

file_names = []
for i in range(0, num_files):
	file_names.append(raw_input("Name of File, including directory (if different from working directory:    "))

predictions = []
for i in range(0, len(file_names)):
	data = pd.read_csv(file_names[i])
	#data = data.groupby('id')
	data = data.sort_index()
	predictions.append( data.drop(['id'], axis=1) )#remove id
	
# memory clean up
ids = np.array(data['id'])
data = None

final_predictions = np.empty(predictions[i].shape)
for prediction in predictions:
	final_predictions += prediction

final_predictions = final_predictions / len(predictions)

# create pandas object for sbmission
submission = pd.DataFrame(index=ids,
                          columns=cols,
                          data=final_predictions)

# write file
submission.to_csv(final_file,index_label='id',float_format='%.3f')