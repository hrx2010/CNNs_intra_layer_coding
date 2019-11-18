import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import slim 
import sys 

# load SLIM source-code package from './slim/'. 
SLIM_PATH = './slim/'
sys.path.append(SLIM_PATH)

from scipy import interpolate 
from scipy.io import savemat 
from nets.vgg import * 
from preprocessing import vgg_preprocessing 
from tools import * 

from inferences import * 
from quantization_methods import *
from transform_methods import *
from pareto_condition_optimization import *
from alexnet_mean_removal import * 

import os
import gc

# command line parameters 
# python generate_RD_curves_vgg16.py A B C
# A: which GPU to run
# B: which layer to process
# C: which kernal to process
# e.g., python generate_RD_curves_vgg16.py 3 1 5 -> run GPU3 to do the statistics for the 5th kernal in layer 1.


# define which GPU is to be run 
gpu_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


# transform method
transform_method = str(sys.argv[2])


# slope rate
slope_rate = int(sys.argv[3])
pareto_condition_slope = -1.0 * np.power(2 , -48 + 0.50*(slope_rate-1))


per_channel_means = [0] * num_conv_layers
convolved_per_channel_means = [0] * num_conv_layers


for i in range(num_conv_layers):
	per_channel_means[i] = np.load('./results_statistics_mean_removal/per_channel_means_%d.npy' % (i))
	convolved_per_channel_means[i] = np.load('./results_statistics_convolved_per_channel_mean/convolved_per_channel_means_%d.npy' % (i))


def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))


#create folder of results
path_results = ('./results_RD_curves_alexnet_mean_removal_%s' % (transform_method)) 
#path_results = ('./results_RD_curves_alexnet_%s' % (transform_method))


if not os.path.exists(path_results):
    os.makedirs(path_results)


# the directory of sample images.
SAMPLES_PATH = './samples/' 
# model name.
model='alexnet' 
# the number of convolutional layers in VGG16. Totally 13 conv layers and 3 fully connected layers in VGG16.
num_conv_layers = 5 
# the path of checkpoint file of pre-trained VGG16 model.
# this checkpoint file is very big (>500 MB). Please download "vgg_16_2016_08_28.tar.gz" from here "https://github.com/tensorflow/models/tree/master/research/slim".
# then extract "vgg_16_2016_08_28.tar.gz" and put it to the same directory of this source file. 
conv_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
checkpoint_file = './bvlc_alexnet.npy' 
# inference flag. If ture, then run inference at the end. Otherwise not.
flag_inference = False


max_rates = 17
num_tesing_images = 1000


# define quantization hyper-parameters
hist_delta = [0] * num_conv_layers
hist_coded = [0] * num_conv_layers
hist_steps = [0] * num_conv_layers
hist_size = [0] * num_conv_layers
hist_filter_dims = [0] * num_conv_layers
INF = 1000000000


# define alex model
dropoutPro = 1
classNum = 1000
skip = []
x = tf.placeholder("float", [1, 227, 227, 3]) 
alexnet_model = alexnet_mean_removal(x, dropoutPro, classNum, skip, per_channel_means_all = per_channel_means, convolved_per_channel_means_all = convolved_per_channel_means)
output_before_softmax = alexnet_model.fc3
scores = tf.nn.softmax(output_before_softmax)


# load alex model
#with tf.Session() as sess:
config = tf.ConfigProto()	
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
alexnet_model.loadModel(sess)


weight_values = np.zeros((num_conv_layers,), dtype=np.object)

total_num_weights = 0

# load RD curves from files
for i in range(num_conv_layers):
	with tf.variable_scope(conv_names[i] , reuse = True):
		node_weights = tf.get_variable('w', trainable = False)

	weight_values_original = sess.run(node_weights)

	[fh, fw, n_input, n_output] = weight_values_original.shape

	total_num_weights = total_num_weights + fh * fw * n_input * n_output

	hist_delta[i] = np.zeros((fh * fw , max_rates))
	hist_coded[i] = np.zeros((fh * fw , max_rates))
	hist_steps[i] = np.zeros((fh * fw , max_rates))
	hist_size[i] = np.zeros((fh * fw , max_rates))
	hist_filter_dims[i] = fh * fw

	file_in = open('%s/RD_curves_layer_%d' % (path_results , i) , "r")

	lines = file_in.readlines()

	for data_per_line in lines:
		print('data_per_line: %s' % (data_per_line))
		values = [a for a in data_per_line.split()]
		dim = int(values[1])
		bits = int(values[2])
		rates = float(values[3])
		errors = float(values[4])
		step = int(values[5])
		q_size = float(values[6])

		hist_delta[i][dim][bits] = errors
		hist_coded[i][dim][bits] = rates
		hist_steps[i][dim][bits] = step
		hist_size[i][dim][bits] = q_size

		print('loading data from file layer %d dimension %d bits %d: rate %f error %f step %d size %f' % (i , dim , bits , rates , errors , step , q_size))

	file_in.close()


# optimize bit allocation in Parato-condition
#total_rate = 1.0 * ave_compressed_rate / 10.0 * total_num_weights
#bit_allocations = pareto_condition_optimization(num_conv_layers , hist_filter_dims , total_rate , max_rates , hist_coded , hist_delta)

bit_allocations, total_rate = pareto_condition_optimization_given_slope(num_conv_layers , hist_filter_dims , max_rates , hist_coded , hist_delta , pareto_condition_slope)

ave_rate = (1.0 * total_rate) / (1.0 * total_num_weights)

for i in range(num_conv_layers):
	for j in range(hist_filter_dims[i]):
		print('bit allocation layer %d dim %d bits %d' % (i , j  , bit_allocations[i][j]))


# run inference of quantized network on ImageNet
for i in range(num_conv_layers):
	with tf.variable_scope(conv_names[i] , reuse = True):
		node_weights = tf.get_variable('w', trainable = False)

	weight_values_original = sess.run(node_weights)
	[fh, fw, n_input, n_output] = weight_values_original.shape

	#weight_values_original_transformed = dft2(weight_values_original)
	#weight_values_original_transformed = transform_given_matrices(transform_matrices[i][0] , weight_values_original)

	if transform_method == 'dft2':
		weight_values_original_transformed = dft2(weight_values_original)
	elif transform_method == 'dst2':
		weight_values_original_transformed = dst2(weight_values_original)
	elif transform_method == 'dct2':
		weight_values_original_transformed = dct2(weight_values_original)
	else:
		sys.exit('no transform method found: %s' % (transform_method))

	
	quantized_weights = np.array(weight_values_original_transformed)

	for j in range(fh * fw):
		r = int((j / fw))
		c = int((j % fw))
		scale = np.floor(np.log2(np.sqrt(np.mean(  np.real(weight_values_original_transformed[r,c,:,:]**2)  ))))
		offset = scale

		allocated_bits = int(bit_allocations[i][j])
		num_step = hist_steps[i][j][allocated_bits]
		#delta = offset + 0.5 * num_step
		#delta = offset + 0.05 * num_step - 2.0
		delta = hist_size[i][j][allocated_bits]
		quantized_weights[r,c,:,:] = quantize(quantized_weights[r,c,:,:] , np.power(2 , delta) , allocated_bits)

	#quantized_weights = np.real(idft2(quantized_weights))
	#quantized_weights = np.real(i_transform_given_matrices(transform_matrices[i][0] , quantized_weights))

	if transform_method == 'dft2':
		quantized_weights = np.real(idft2(quantized_weights))
	elif transform_method == 'dst2':
		quantized_weights = np.real(idst2(quantized_weights))
	elif transform_method == 'dct2':
		quantized_weights = np.real(idct2(quantized_weights))
	else:
		sys.exit('no transform method found: %s' % (transform_method))

	sess.run(node_weights.assign(quantized_weights))


top_1, top_5 = run_inference_alexnet(sess, x, scores, number_validation_images=num_tesing_images) 

file_results = open(("accuracy_alexnet_mean_removal_%s.txt" % (transform_method)), "a+")

file_results.write("%f %f %f\n" % (ave_rate , top_1 , top_5))

file_results.close()

print('top1 %f top5 %f' % (top_1 , top_5))
	
