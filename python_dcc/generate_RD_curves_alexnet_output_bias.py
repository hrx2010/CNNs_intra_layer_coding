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

import os
import gc

# command line parameters 
# python generate_RD_curves_vgg16.py A B C
# A: which GPU to run
# B: which layer to process
# C: which kernal to process
# e.g., python generate_RD_curves_vgg16.py 3 1 5 -> run GPU3 to do the statistics for the 5th kernal in layer 1.


gpu_id_str = sys.argv[1]
gpu_id_int = int(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str


# number of total gpus on server
processing_layer_id = int(sys.argv[2])

# number of dimension
processing_dim_id = int(sys.argv[3])

def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))


#create folder of results
path_results = './results_RD_curves_alexnet_dft2' 
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
# define_transform_method
transform_method = 'dft2'

print('asdasdasd %d' % (len(conv_names)))

# define quantization hyper-parameters
max_steps = 50
max_rates = 11
hist_delta = [0] * num_conv_layers
hist_coded = [0] * num_conv_layers
hist_steps = [0] * num_conv_layers
hist_size = [0] * num_conv_layers
INF = 1000000000


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

file_results = open('%s/RD_curves_layer_%d' % (path_results , processing_layer_id) , "a+")




# do the statistics for each conv layer.
for i in range(num_conv_layers):
	if i != processing_layer_id:
		continue

	with tf.variable_scope(conv_names[i] , reuse = True):
		node_weights = tf.get_variable('w', trainable = False)
		node_bias = tf.get_variable('b', trainable = False)
		node_bias_values = sess.run(node_bias)
		print(node_bias_values)

	weight_values_original = sess.run(node_weights)
	#np_weights_values_original = np.array(weight_values_original)

	[fh, fw, n_input, n_output] = weight_values_original.shape

	hist_delta[i] = np.zeros((fh * fw , max_rates))
	hist_coded[i] = np.zeros((fh * fw , max_rates))
	hist_steps[i] = np.zeros((fh * fw , max_rates))
	hist_size[i] = np.zeros((fh * fw , max_rates))

	if transform_method == 'dft2':
		weight_values_original_transformed = dft2(weight_values_original)
	elif transform_method == 'dst2':
		weight_values_original_transformed = dst2(weight_values_original)
	elif transform_method == 'dct2':
		weight_values_original_transformed = dct2(weight_values_original)
	else:
		sys.exit('no transform method found: %s' % (transform_method))

	weight_values_original_transformed = np.array(weight_values_original_transformed)

	for j in range(fh * fw):
		if j != processing_dim_id:
			continue

		r = int((j / fw))
		c = int((j % fw))
		scale = np.floor(np.log2(np.sqrt(np.mean(  np.real(weight_values_original_transformed[r,c,:,:]**2)  ))))
		coded = INF
		offset = scale + 2
		
		for k in range(max_rates):
			B = k

			hist_delta[i][j][k] = -1
			hist_coded[i][j][k] = -1
			hist_steps[i][j][k] = -1
			hist_size[i][j][k] = -1

			min_output_error = INF
			optimized_t = INF

			last_output_error = INF
			last_mse = INF

			for t in range(max_steps):
				delta = offset + 0.25 * t
				#delta = (offset + 0.005 * t - 2.0)
				
				quantized_weights = np.array(weight_values_original_transformed)
				quantized_weights[r,c,:,:] = quantize(quantized_weights[r,c,:,:] , np.power(2 , delta) , B) #B
				coded = fixed_length_entropy(quantized_weights[r,c,:,:] , B) * n_input * n_output #B

				if transform_method == 'dft2':
					inverted_quantized_weights = np.real(idft2(quantized_weights))
				elif transform_method == 'dst2':
					inverted_quantized_weights = np.real(idst2(quantized_weights))
				elif transform_method == 'dct2':
					inverted_quantized_weights = np.real(idct2(quantized_weights))
				else:
					sys.exit('no transform method found: %s' % (transform_method))

				#set_variable_to_tensor(sess , vgg_weights[i] , inverted_quantized_weights)
				sess.run(node_weights.assign(inverted_quantized_weights))
				last_layer_output_quantized = run_inference_alexnet_last_layer_output(sess, x, output_before_softmax)
				last_layer_output_quantized = np.array(last_layer_output_quantized)

				cur_output_error = np.mean((last_layer_output - last_layer_output_quantized)**2)
				cur_mse = np.mean((weight_values_original - inverted_quantized_weights)**2)

				if t == 0 or cur_output_error < min_output_error:
					min_output_error = cur_output_error
					hist_delta[i][j][k] = cur_output_error
					hist_coded[i][j][k] = coded
					hist_steps[i][j][k] = t
					hist_size[i][j][k] = delta

				if t > 0 and last_output_error < cur_output_error and last_mse < cur_mse:
					offset = delta - 2
					break

				print('layer %d , kernal (%d , %d) [%d , %d], rate %d bits, step %d, delta %f, total rate %f, output error %.14f (min error %.14f)' % (i , r , c , fh , fw , k , t , delta , coded , cur_output_error , hist_delta[i][j][k]))

				if k == 0:
					break
				
				last_output_error = cur_output_error
				last_mse = cur_mse
			
			file_results.write("%d %d %d %f %.14f %d %.10f\n" % (i , j , k , hist_coded[i][j][k] , hist_delta[i][j][k] , hist_steps[i][j][k] , hist_size[i][j][k]))

			gc.collect()

		#set_variable_to_tensor(sess , vgg_weights[i] , weight_values_original)

file_results.close()
	
