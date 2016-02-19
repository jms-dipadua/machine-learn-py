import numpy as np

def scale_data(x_vals):
	print len(x_vals[0,:])
	for i in range(1, len(x_vals[:,0])):
		# avoid exceeding length of array (so it don't break)
		if i > len(x_vals[0,:]):
			break
		# get the min_max (for range) and then scale
		else:
			mean = np.mean(x_vals[:, i-1])
			#print mean
			x_vals[:, i-1] = x_vals[:, i-1] - mean
			range_val = np.amax(x_vals[:, i-1]) - np.amin(x_vals[:, i-1])
			x_vals[:, i-1] = x_vals[:, i-1] / range_val
	return x_vals