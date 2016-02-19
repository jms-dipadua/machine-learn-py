# data loading and general matrix stuff
import numpy as np
import pandas as pd
from glob import glob
import sys 
from collections import defaultdict
import csv



def prepare_scurb_file(fname):
    """ read and scurb file """
    # Read data
    data = pd.read_csv(fname)
    #remove a BUNCH of columns
    clean=data.drop(['VAR_0008', 'VAR_0009', 'VAR_0010', 'VAR_0011', 'VAR_0012', 'VAR_0019', 'VAR_0020', 'VAR_0021', 'VAR_0022', 'VAR_0023', 'VAR_0024', 'VAR_0025', 'VAR_0026', 'VAR_0027', 'VAR_0028', 'VAR_0029', 'VAR_0030', 'VAR_0031', 'VAR_0032', 'VAR_0038', 'VAR_0039', 'VAR_0040', 'VAR_0041', 'VAR_0042', 'VAR_0043', 'VAR_0044', 'VAR_0045', 'VAR_0073', 'VAR_0074', 'VAR_0075', 'VAR_0098', 'VAR_0099', 'VAR_0100', 'VAR_0114', 'VAR_0115', 'VAR_0116', 'VAR_0130', 'VAR_0131', 'VAR_0132', 'VAR_0138', 'VAR_0139', 'VAR_0140', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0197', 'VAR_0200', 'VAR_0202', 'VAR_0204', 'VAR_0205', 'VAR_0206', 'VAR_0207', 'VAR_0213', 'VAR_0214', 'VAR_0215', 'VAR_0216', 'VAR_0217', 'VAR_0226', 'VAR_0227', 'VAR_0228', 'VAR_0229', 'VAR_0230', 'VAR_0237', 'VAR_0239', 'VAR_0246', 'VAR_0270', 'VAR_0274', 'VAR_0275', 'VAR_0276', 'VAR_0277', 'VAR_0278', 'VAR_0313', 'VAR_0314', 'VAR_0315', 'VAR_0316', 'VAR_0317', 'VAR_0318', 'VAR_0319', 'VAR_0320', 'VAR_0321', 'VAR_0367', 'VAR_0394', 'VAR_0395', 'VAR_0396', 'VAR_0397', 'VAR_0398', 'VAR_0399', 'VAR_0404', 'VAR_0411', 'VAR_0412', 'VAR_0413', 'VAR_0414', 'VAR_0415', 'VAR_0467', 'VAR_0493', 'VAR_0531', 'VAR_0546', 'VAR_0547', 'VAR_0548', 'VAR_0549', 'VAR_0551', 'VAR_0570', 'VAR_0574', 'VAR_0575', 'VAR_0576', 'VAR_0577', 'VAR_0598', 'VAR_0599', 'VAR_0600', 'VAR_0601', 'VAR_0602', 'VAR_0603', 'VAR_0632', 'VAR_0633', 'VAR_0634', 'VAR_0635', 'VAR_0636', 'VAR_0637', 'VAR_0638', 'VAR_0639', 'VAR_0640', 'VAR_0641', 'VAR_0642', 'VAR_0643', 'VAR_0644', 'VAR_0645', 'VAR_0653', 'VAR_0654', 'VAR_0659', 'VAR_0660', 'VAR_0669', 'VAR_0670', 'VAR_0671', 'VAR_0672', 'VAR_0673', 'VAR_0674', 'VAR_0675', 'VAR_0676', 'VAR_0677', 'VAR_0678', 'VAR_0679', 'VAR_0680', 'VAR_0681', 'VAR_0682', 'VAR_0684', 'VAR_0691', 'VAR_0692', 'VAR_0693', 'VAR_0694', 'VAR_0695', 'VAR_0696', 'VAR_0697', 'VAR_0698', 'VAR_0699', 'VAR_0700', 'VAR_0701', 'VAR_0702', 'VAR_0703', 'VAR_0710', 'VAR_0714', 'VAR_0732', 'VAR_0734', 'VAR_0735', 'VAR_0745', 'VAR_0755', 'VAR_0757', 'VAR_0763', 'VAR_0773', 'VAR_0779', 'VAR_0784', 'VAR_0789', 'VAR_0798', 'VAR_0799', 'VAR_0803', 'VAR_0804', 'VAR_0808', 'VAR_0809', 'VAR_0811', 'VAR_0840', 'VAR_0851', 'VAR_0855', 'VAR_0856', 'VAR_0857', 'VAR_0858', 'VAR_0862', 'VAR_0865', 'VAR_0873', 'VAR_0882', 'VAR_0883', 'VAR_0889', 'VAR_0890', 'VAR_0891', 'VAR_0901', 'VAR_0902', 'VAR_0903', 'VAR_0904', 'VAR_0905', 'VAR_0906', 'VAR_0907', 'VAR_0908', 'VAR_0909', 'VAR_0910', 'VAR_0911', 'VAR_0912', 'VAR_0913'], axis=1)
    print np.shape(clean)
    raw_input('PRESS ENTER TO CONTINUE')
    data = None
    return  clean

def write_scurb_file(scrub_file, scrubbed_data):
	# write file
	scrubbed_data.to_csv(scrub_file,index_label='id',float_format=None)


def initialize ():
	# 1. prompt for file names
	# 2. scrub file 
	# 3. write to new file 

	# 1. file names
	read_file = raw_input("What is the File to READ AND SCRUB?")
	print ("Great. Next the output file.")
	scrub_file = raw_input("What is the OUTPUT file name?")
	
	# 2. scrub file
	scrubbed_data = prepare_scurb_file(read_file)

	# 3. write new file
	final_file = write_scurb_file(scrub_file, scrubbed_data)
	if final_file:
		print "final file saved:  %r" % final_file


initialize()