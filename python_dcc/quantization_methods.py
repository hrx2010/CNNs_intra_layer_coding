import numpy as np
import math

def quantize(x , q_delta , q_bits):
	#print('quantize bits %d' % (q_bits))

	if q_bits == 0:
		return x * 0.0

	if q_bits != 0:
		min_point = -(np.power(2 , (q_bits - 1))) * q_delta
		max_point = (np.power(2 , (q_bits - 1))) * q_delta - q_delta
	else:
		min_point = 0
		max_point = 0

	#x = np.array(x)
	
	# change np.floor to np.around
	# x_hat = np.clip(q_delta * np.floor(x / q_delta + 0.5) , min_point , max_point)
	x_hat = np.clip(q_delta * np.around(x / q_delta) , min_point , max_point)

	#print(x_hat)

	return x_hat

def quantize_based_on_q_level(x , q_delta , q_level):
	#print('quantize bits %d' % (q_bits))

	if q_level == 0:
		return x * 0.0

	if q_level != 0:
		min_point = -(1.0 * q_level / 2.0) * q_delta
		max_point = (1.0 * q_level / 2.0) * q_delta
	else:
		min_point = 0
		max_point = 0

	#x = np.array(x)
	
	# change np.floor to np.around
	# x_hat = np.clip(q_delta * np.floor(x / q_delta + 0.5) , min_point , max_point)
	x_hat = np.clip(q_delta * np.around(x / q_delta) , min_point , max_point)

	#print(x_hat)

	return x_hat

def fixed_length_entropy(symbols , B):
	return B

