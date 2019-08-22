import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

import gc

# command line parameters 
# python generate_RD_curves_vgg16.py A B C
# A: which GPU to run
# B: which transform to use
# C: number of images to use
# D: which layer to run
# e.g., python generate_RD_curves_vgg16.py 3 1 5 -> run GPU3 to do the statistics for the 5th kernal in layer 1.


# number of total gpus on server
gpu_id_str = sys.argv[1]
gpu_id_int = int(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str

tranname = sys.argv[2]

# number of images
max_images = int(sys.argv[3])

# which layer to run
processing_layer_id = int(sys.argv[4])


def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))


#create folder of results
path_results = './results_RD_curves_VGG_16' 
if not os.path.exists(path_results):
    os.makedirs(path_results)


# the directory of sample images.
SAMPLES_PATH = '/home/wangzhe/Documents/data/ImageNet2012/val/'#'./samples/' 
# model name.
model='vgg_16' 
# the number of convolutional layers in VGG16. Totally 13 conv layers and 3 fully connected layers in VGG16.
num_conv_layers = 13 
# the path of checkpoint file of pre-trained VGG16 model.
# this checkpoint file is very big (>500 MB). Please download "vgg_16_2016_08_28.tar.gz" from here "https://github.com/tensorflow/models/tree/master/research/slim".
# then extract "vgg_16_2016_08_28.tar.gz" and put it to the same directory of this source file. 
checkpoint_file = './vgg_16.ckpt' 
# inference flag. If ture, then run inference at the end. Otherwise not.
flag_inference = False


# define quantization hyper-parameters
max_steps = 50
max_rates = 17

hist_y_mse = [0] * num_conv_layers
hist_w_mse = [0] * num_conv_layers
hist_coded = [0] * num_conv_layers
hist_steps = [0] * num_conv_layers
# INF = 1000000000

# transform functions

tran = transforms[tranname]
# create tensorflow graph of VGG16 
with slim.arg_scope(vgg_arg_scope()):
	# define the tensor of input image .
	input_string = tf.placeholder(tf.string)
	# preprocess the input image - JPEG decoding, resizing, etc.
	processed_images = tensor_preprocessed_input_images(input_string)
	# define VGG16 model. 'vgg_activations' contains all the intermediate outputs of conv layers before RELU.
	logits, _ = vgg_16_original(processed_images, num_classes=1000, is_training=False)
	# compute prediction scores for image classification 
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


# load all the samples of the input images
# input_images = read_samples_from_file('list_samples.txt')


# extract all conv layers' weights. vgg_weights[i] denotes the weights of i-th conv layer. 
vgg_weights = get_all_weights_variables(variables_to_restore)

weight_values = np.zeros((num_conv_layers,), dtype=np.object)

last_layer_output = run_inference_VGG16_last_layer_output(sess, input_string, logits, max_images)
last_layer_output = np.array(last_layer_output)

file_results = open('%s/RD_curves_layer_%d' % (path_results , processing_layer_id) , "a+")


# do the statistics for each conv layer.
for i in range(num_conv_layers):
	if i != processing_layer_id:
		continue

	weight_values_original = sess.run(vgg_weights[i])

	[fh, fw, n_input, n_output] = weight_values_original.shape

	hist_y_mse[i] = np.zeros((fh * fw , max_rates))
	hist_w_mse[i] = np.zeros((fh * fw , max_rates))
	hist_coded[i] = np.zeros((fh * fw , max_rates))
	hist_steps[i] = np.zeros((fh * fw , max_rates))

	weight_values_original_transformed = tran[0](weight_values_original)
	weight_values_original_transformed = np.array(weight_values_original_transformed)

	for j in range(fh * fw):
		(r,c) = np.unravel_index(j,(fh,fw));
		scale = np.floor(np.log2(np.sqrt(np.mean(  np.real(weight_values_original_transformed[r,c,:,:]**2)  ))))
		coded = np.Inf
		offset = scale + 2
		
		for k in range(max_rates):
			B = k

			hist_y_mse[i][j][k] = np.NaN
			hist_coded[i][j][k] = np.NaN
			hist_steps[i][j][k] = np.NaN

			best_output_error = np.Inf
			best_weight_error = np.Inf
			prev_output_error = np.Inf
			prev_weight_error = np.Inf
			best_weight_delta = np.NaN

			for t in range(max_steps):
				delta = offset + 0.25 * t
				#delta = (offset + 0.005 * t - 2.0)
				
				quantized_weights = np.array(weight_values_original_transformed)
				quantized_weights[r,c,:,:] = quantize(quantized_weights[r,c,:,:] , np.power(2 , delta) , B) #B
				coded = fixed_length_entropy(quantized_weights[r,c,:,:] , B) * n_input * n_output #B

				inverted_quantized_weights = tran[1](quantized_weights)
				
				set_variable_to_tensor(sess , vgg_weights[i] , inverted_quantized_weights)
				last_layer_output_quantized = run_inference_VGG16_last_layer_output(sess, input_string, logits, max_images)
				last_layer_output_quantized = np.array(last_layer_output_quantized)

				curr_output_error = np.mean((last_layer_output - last_layer_output_quantized)**2)
				curr_weight_error = np.mean((weight_values_original - inverted_quantized_weights)**2)

				print('%s %s | layer: %03d/%03d, band: %03d/%03d, scale: %+6.2f, delta: %+6.2f, ymse: %5.2e, wmse: %5.2e, rate: %5.2e' % (model, tranname, i , num_conv_layers, j, fh*fw , scale, delta , curr_output_error , curr_weight_error, k))

				if curr_output_error < best_output_error:
					best_output_error = curr_output_error
					best_weight_error = curr_weight_error
					best_weight_delta = delta
				if curr_output_error > prev_output_error and curr_weight_error > prev_weight_error or k == 0:
					offset = best_weight_delta - 2
					break
				prev_output_error = curr_output_error
				prev_weight_error = curr_weight_error

			hist_y_mse[i][j][k] = best_output_error
			hist_w_mse[i][j][k] = best_weight_error
			hist_steps[i][j][k] = best_weight_delta
			hist_coded[i][j][k] = coded
			
			file_results.write("%d %d %d %f %.10f %d\n" % (i , j , k , hist_coded[i][j][k] , hist_y_mse[i][j][k] , hist_steps[i][j][k]))

			gc.collect()

		set_variable_to_tensor(sess , vgg_weights[i] , weight_values_original)

file_results.close()
	
