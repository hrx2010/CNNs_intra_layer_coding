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

import os


def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))


# define which GPU is to be run 
gpu_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


# define the average rate of compressed network
ave_compressed_rate = int(sys.argv[2])


# define model path 
SAMPLES_PATH = './samples/' 
model='vgg_16'  
checkpoint_file = './vgg_16.ckpt' 


# define quantization hyper-parameters
num_conv_layers = 13
max_rates = 17
INF = 1000000000
total_num_weights = 0
num_tesing_images = 1000

#define variables of RD curves
hist_delta = [0] * num_conv_layers
hist_coded = [0] * num_conv_layers
hist_steps = [0] * num_conv_layers
hist_filter_dims = [0] * num_conv_layers


# create tensorflow graph of VGG16 
with slim.arg_scope(vgg_arg_scope()):
	input_string = tf.placeholder(tf.string)
	processed_images = tensor_preprocessed_input_images(input_string)
	logits, _, vgg_activations, vgg_activations_after_relu = vgg_16_decomposed(processed_images, num_classes=1000, is_training=False)
	probabilities = tf.nn.softmax(logits)


# create tensorflow session
config = tf.ConfigProto()	
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# load parameters from the checkpoint file (vgg_16.ckpt)
variables_to_restore = slim.get_variables_to_restore()
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
init_fn(sess)


# extract all conv layers' weights. vgg_weights[i] denotes the weights of i-th conv layer. 
vgg_weights = get_all_weights_variables(variables_to_restore)


# load RD curves from files
for i in range(num_conv_layers):
	weight_values_original = sess.run(vgg_weights[i])
	[fh, fw, n_input, n_output] = weight_values_original.shape
	total_num_weights = total_num_weights + fh * fw * n_input * n_output

	hist_delta[i] = np.zeros((fh * fw , max_rates))
	hist_coded[i] = np.zeros((fh * fw , max_rates))
	hist_steps[i] = np.zeros((fh * fw , max_rates))
	hist_filter_dims[i] = fh * fw

	file_in = open('RD_curves_layer_%d' % (i) , "r")

	lines = file_in.readlines()

	for x in lines:
		print('x: %s' % (x))
		values = [a for a in x.split()]
		dim = int(values[1])
		bits = int(values[2])
		rates = float(values[3])
		errors = float(values[4])
		step = int(values[5])

		hist_delta[i][dim][bits] = errors
		hist_coded[i][dim][bits] = rates
		hist_steps[i][dim][bits] = step

		print('loading data from file layer %d dimension %d bits %d: rate %f error %f step %d' % (i , dim , bits , rates , errors , step))

	file_in.close()


# optimize bit allocation in Parato-condition
total_rate = 1.0 * ave_compressed_rate / 10.0 * total_num_weights
bit_allocations = pareto_condition_optimization(num_conv_layers , hist_filter_dims , total_rate , max_rates , hist_coded , hist_delta)

#for i in range(num_conv_layers):
#	for j in range(hist_filter_dims[i]):
#		print('bit allocation layer %d dim %d bits %d' % (i , j  , bit_allocations[i][j]))

# run inference of quantized network on ImageNet
for i in range(num_conv_layers):
	weight_values_original = sess.run(vgg_weights[i])
	[fh, fw, n_input, n_output] = weight_values_original.shape
	weight_values_original_transformed = dft2(weight_values_original)
	quantized_weights = np.array(weight_values_original_transformed)

	for j in range(fh * fw):
		r = int((j / fw))
		c = int((j % fw))
		scale = np.floor(np.log2(np.sqrt(np.mean(  np.real(weight_values_original_transformed[r,c,:,:]**2)  ))))
		offset = scale

		allocated_bits = int(bit_allocations[i][j])
		num_step = hist_steps[i][j][allocated_bits]
		delta = offset + 0.5 * num_step

		quantized_weights[r,c,:,:] = quantize(quantized_weights[r,c,:,:] , np.power(2 , delta) , allocated_bits)

	quantized_weights = np.real(idft2(quantized_weights))
	set_variable_to_tensor(sess , vgg_weights[i] , quantized_weights)

top_1, top_5 = run_inference_VGG16(sess, input_string, probabilities, number_validation_images=num_tesing_images)

print('top1 %f top5 %f' % (top_1 , top_5))
	
