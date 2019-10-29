import numpy as np 

eps_slope = 1e-10
INF = 1e12
max_itr_times = 50

def pareto_condition_optimization_given_slope(num_conv_layers , hist_filter_dims  , num_quant_levels , hist_coded , hist_delta , val_slope):

	#solutions = [0] * num_conv_layers
	bit_allocations = [0] * num_conv_layers

	cur_slope = val_slope

	total_compressed_size = 0.0
	total_output_error = 0.0
		
	for i in range(num_conv_layers):
		bit_allocations[i] = np.zeros((hist_filter_dims[i]))

		for j in range(hist_filter_dims[i]):
			min_intercept = INF
			allocated_bits = -1

			for k in range(num_quant_levels):
				print('i j k %d %d %d (dim %d)' % (i , j , k , hist_filter_dims[i]))
				cur_R = hist_coded[i][j][k]
				cur_D = hist_delta[i][j][k]
				cur_intercept = cur_D - cur_slope * cur_R

				print("i %d j %d k %d min_intercept %.12f cur intercept %.12f slope %.12f" % (i , j , k , min_intercept , cur_intercept , cur_slope))

				if (cur_intercept < min_intercept or k == 0):
					min_intercept = cur_intercept
					allocated_bits = k

			bit_allocations[i][j] = allocated_bits
			total_compressed_size += hist_coded[i][j][allocated_bits]



	return bit_allocations, total_compressed_size

def pareto_condition_optimization(num_conv_layers , hist_filter_dims , total_rate , num_quant_levels , hist_coded , hist_delta):
	left_bound = -1e12
	right_bound = 0.0

	solutions = [0] * num_conv_layers
	bit_allocations = [0] * num_conv_layers

	count_itr = 0

	while (right_bound - left_bound > eps_slope):
		mid = (left_bound + right_bound) / 2
		cur_slope = mid
		count_itr = count_itr + 1

		total_compressed_size = 0.0
		total_output_error = 0.0

		

		for i in range(num_conv_layers):
			bit_allocations[i] = np.zeros((hist_filter_dims[i]))

			for j in range(hist_filter_dims[i]):
				min_intercept = INF
				allocated_bits = -1

				for k in range(num_quant_levels):
					cur_R = hist_coded[i][j][k]
					cur_D = hist_delta[i][j][k]
					cur_intercept = cur_D - cur_slope * cur_R

					if (cur_intercept < min_intercept or k == 0):
						min_intercept = cur_intercept
						allocated_bits = k

				bit_allocations[i][j] = allocated_bits
				total_compressed_size += hist_coded[i][j][allocated_bits]

		if total_compressed_size < total_rate:
			for i in range(num_conv_layers):
				solutions[i] = bit_allocations[i]

		if total_compressed_size < total_rate:
			left_bound = mid
		else:
			right_bound = mid

		print('left %f right %f: mid %.12f (total %f max %f)' % (left_bound , right_bound , mid , total_compressed_size , total_rate))

	return solutions


def pareto_condition_optimization_layer_wise(num_conv_layers , hist_num_weights , total_rate , num_quant_levels , hist_coded , hist_delta):
	left_bound = -1e12
	right_bound = 0.0

	solutions = [0] * num_conv_layers
	bit_allocations = [0] * num_conv_layers

	count_itr = 0

	while (right_bound - left_bound > eps_slope):
		mid = (left_bound + right_bound) / 2
		cur_slope = mid
		count_itr = count_itr + 1

		#print('left %f right %f: mid %f' % (left_bound , right_bound , mid))

		total_compressed_size = 0.0
		total_output_error = 0.0

		for i in range(num_conv_layers):
			#bit_allocations[i] = np.zeros((hist_filter_dims[i]))
			#for j in range(hist_filter_dims[i]):

				min_intercept = INF
				allocated_bits = -1

				for k in range(num_quant_levels):
					cur_R = hist_coded[i][k]
					cur_D = hist_delta[i][k]
					cur_intercept = cur_D - cur_slope * cur_R

					if (cur_intercept < min_intercept or k == 0):
						min_intercept = cur_intercept
						allocated_bits = k

				bit_allocations[i] = allocated_bits
				total_compressed_size += hist_coded[i][allocated_bits]

		if total_compressed_size < total_rate:
			for i in range(num_conv_layers):
				solutions[i] = bit_allocations[i]

		if total_compressed_size < total_rate:
			left_bound = mid
		else:
			right_bound = mid

	return solutions

				
