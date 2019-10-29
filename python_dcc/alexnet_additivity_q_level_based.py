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
import gc


# define which GPU is to be run 
gpu_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


# define the quantization bits and quantization layers
quantization_level = int(sys.argv[2])
quantization_layers = int(sys.argv[3])


def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))


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
# define_transform_method
transform_method = 'dft2'
max_rates = 11
num_tesing_images = 1000
max_steps = 50


# define alex model
dropoutPro = 1
classNum = 1000
skip = []
x = tf.placeholder("float", [1, 227, 227, 3]) 
alexnet_model = alexnet.alexNet(x, dropoutPro, classNum, skip)
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

last_layer_output = run_inference_alexnet_last_layer_output(sess, x, output_before_softmax)
last_layer_output = np.array(last_layer_output)


all_weights_values_original = [0] * num_conv_layers
for i in range(num_conv_layers):
	with tf.variable_scope(conv_names[i] , reuse = True):
		node_weights = tf.get_variable('w', trainable = False)
	
	all_weights_values_original[i] = sess.run(node_weights)


optimal_delta_layers = [0] * num_conv_layers
estimated_output_error = 0


# do the statistics for each conv layer.
for i in range(num_conv_layers):
	if ((quantization_layers & (1 << i)) == 0):
		continue

	print('i %d' % (i))

	with tf.variable_scope(conv_names[i] , reuse = True):
		node_weights = tf.get_variable('w', trainable = False)

	scale = np.floor(np.log2(np.sqrt(np.mean(  np.real(all_weights_values_original[i]**2)  ))))
	offset = scale - 6

	min_output_error = INF
	pre_output_error = INF
	pre_mse = INF

	for t in range(max_steps):
		delta = offset + 0.25 * t
		quantized_weights = np.array(all_weights_values_original[i])
		quantized_weights = quantize_based_on_q_level(quantized_weights , np.power(2 , delta) , quantization_level) #B
		sess.run(node_weights.assign(quantized_weights))
		last_layer_output_quantized = run_inference_alexnet_last_layer_output(sess, x, output_before_softmax)
		last_layer_output_quantized = np.array(last_layer_output_quantized)
		
		cur_output_error = np.mean((last_layer_output - last_layer_output_quantized)**2)
		cur_mse = np.mean((all_weights_values_original[i] - quantized_weights)**2)

		if t == 0 or cur_output_error < min_output_error:
			min_output_error = cur_output_error
			optimal_delta_layers[i] = delta

		if pre_output_error < cur_output_error and pre_mse < cur_mse:
			break

		pre_output_error = cur_output_error
		pre_mse = cur_mse

	print('min output error %f' % (min_output_error))

	sess.run(node_weights.assign(all_weights_values_original[i]))

	estimated_output_error += min_output_error


# do the statistics for quantizing multiple layers
for i in range(num_conv_layers):
	if ((quantization_layers & (1 << i)) == 0):
		continue

	print('i %d' % (i))

	with tf.variable_scope(conv_names[i] , reuse = True):
		node_weights = tf.get_variable('w', trainable = False)

	delta = optimal_delta_layers[i]
	quantized_weights = np.array(all_weights_values_original[i])
	quantized_weights = quantize_based_on_q_level(quantized_weights , np.power(2 , delta) , quantization_level)
	sess.run(node_weights.assign(quantized_weights))

last_layer_output_quantized = run_inference_alexnet_last_layer_output(sess, x, output_before_softmax)
last_layer_output_quantized = np.array(last_layer_output_quantized)
true_output_error = np.mean((last_layer_output - last_layer_output_quantized)**2)


file_results = open(('alexnet_additivity_%d_q_level.txt' % (quantization_level)), "a+")

file_results.write("%d %d %.16f %.16f\n" % (quantization_level , quantization_layers , estimated_output_error , true_output_error))

file_results.close()

print('%d %d %.16f %.16f' % (quantization_level , quantization_layers , estimated_output_error , true_output_error))
	
