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

def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))

# the directory of sample images.
SAMPLES_PATH = './samples/' 
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
max_steps = 32
max_rates = 17
hist_delta = [0] * num_conv_layers
hist_coded = [0] * num_conv_layers
INF = 1000000000

# create tensorflow graph of VGG16 
with slim.arg_scope(vgg_arg_scope()):
	# define the tensor of input image .
	input_string = tf.placeholder(tf.string)
	# preprocess the input image - JPEG decoding, resizing, etc.
	processed_images = tensor_preprocessed_input_images(input_string)
	# define VGG16 model. 'vgg_activations' contains all the intermediate outputs of conv layers before RELU.
	logits, _, vgg_activations, vgg_activations_after_relu = vgg_16_decomposed(processed_images, num_classes=1000, is_training=False)
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
input_images = read_samples_from_file('list_samples.txt')

# extract all conv layers' weights. vgg_weights[i] denotes the weights of i-th conv layer. 
vgg_weights = get_all_weights_variables(variables_to_restore)

weight_values = np.zeros((num_conv_layers,), dtype=np.object)

last_layer_output = run_inference_VGG16_last_layer_output(sess, input_string, logits)
last_layer_output = np.array(last_layer_output)

# do the statistics for each conv layer.
for i in range(num_conv_layers):
	#i = 0

	# get the tensor of the weights in conv layer i.  
	weight_values_original = sess.run(vgg_weights[i])

	# get the dimensions of the tensor where fh * fw denotes the kernel size, 
	# n_input denotes the number of inputs and n_output denotes the number of outputs.
	[fh, fw, n_input, n_output] = weight_values_original.shape

	hist_delta[i] = np.zeros((max_rates , max_steps , fh * fw))
	hist_coded[i] = np.zeros((max_rates , max_steps , fh * fw))

	weight_values_original_transformed = dft2(weight_values_original)
	weight_values_original_transformed = np.array(weight_values_original_transformed)

	for j in range(fh * fw):
		r = int((j / fw))
		c = int((j % fw))
		scale = np.floor(np.log2(np.sqrt(np.mean(  np.real(weight_values_original_transformed[r,c,:,:]**2)  ))))
		coded = INF
		offset = scale
		
		for k in range(max_rates):
			#B = k
			B = k

			#print('layer i %d Bit B %d' % (i , B))

			for t in range(max_steps):
				delta = offset + 0.5 * t

				quantized_weights = np.array(weight_values_original_transformed)

				#print(quantized_weights[r,c,:,:])

				#print('delta %f B %f' % (delta , B))
				quantized_weights[r,c,:,:] = quantize(quantized_weights[r,c,:,:] , np.power(2 , delta) , B) #B
				coded = fixed_length_entropy(quantized_weights[r,c,:,:] , B) * n_input * n_output #B

				#print('-----------------------')

				#print(quantized_weights[r,c,:,:])

				quantized_weights = np.real(idft2(quantized_weights))
				
				set_variable_to_tensor(sess , vgg_weights[i] , quantized_weights)
				last_layer_output_quantized = run_inference_VGG16_last_layer_output(sess, input_string, logits)
				last_layer_output_quantized = np.array(last_layer_output_quantized)
				
				hist_coded[i][k][t][j] = coded
				hist_delta[i][k][t][j] = np.mean((last_layer_output - last_layer_output_quantized)**2)
				
				print('layer %d , kernal (%d , %d), rate %d bits, step %d, total rate %f, output error %f' % (i , r , c , k , t , coded , hist_delta[i][k][t][j]))

				set_variable_to_tensor(sess , vgg_weights[i] , weight_values_original)
		
	
