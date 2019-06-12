import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import slim
import sys 

# load SLIM source-code package from './slim/'. 
SLIM_PATH = './slim/'
sys.path.append(SLIM_PATH)

from nets.vgg import *
from preprocessing import vgg_preprocessing 
from tools import *
from inferences import *

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
flag_inference = True

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


# do the statistics for each conv layer.
for i in range(num_conv_layers):

	# get the tensor of the weights in conv layer i.  
	weight_values = sess.run(vgg_weights[i])
	# get the dimensions of the tensor where fh * fw denotes the kernel size, n_input denotes the number of inputs and n_output denotes the number of outputs.
	[fh, fw, n_input, n_output] = weight_values.shape

	# calculate the statistical items of weights for each input.
	for j in range(n_input):
		# extract weights of input j.
		H = weight_values[:,:,j,:]
		# calculate R_H.
		R_H = np.mean(np.square(np.abs(np.fft.fft2(H,axes=(0,1)))), axis=2)
		# output L_H
		print('layer %d, n_input %d, LH = ' % (i , j))
		print(R_H)		

	# define the statistical items of activations.
	activations_mean = 0.0 
	activations_variance = 0.0
	[b, h, w, d] = vgg_activations[i].shape
	activations_PSD = np.zeros(shape=(h, w))

	# calculate the statistical items of activations
	for j in range(len(input_images)):
		# get the path of input image 
		img_path = SAMPLES_PATH + input_images[j] 
		
		# get the activations of the i-th conv layer before RELU
		# for the first layer, input tensor is input image
		input_tensor = input_string
		input_tensor_value = img_path

		# for layer i (i > 0), input tensor is (i-1)th layer's output (after relu) 
		if (i > 0): 
			input_tensor = vgg_activations_after_relu[i-1]
			input_tensor_value = sess.run(vgg_activations_after_relu[i-1] , feed_dict={input_string:img_path})

		# obtain activations (before relu) in this layer
		activation_values = sess.run(vgg_activations[i] , feed_dict={input_tensor:input_tensor_value})
		activation_values_reduced = np.squeeze(activation_values , axis=0)
		
		# get the dimensions of activations
		[h, w, d] = activation_values_reduced.shape
	
		# compute the statistical items for each output
		for k in range(n_output):
			sub_arrays = activation_values_reduced[:,:,k]
			activations_mean += np.mean(sub_arrays)
			activations_variance += np.var(sub_arrays)
			activations_PSD += np.square(np.absolute(np.fft.fft2(sub_arrays)))

	# calculate mean, variance and PSD
	activations_mean = activations_mean / (float(int(n_output))) / (float(len(input_images)))
	activations_variance = activations_variance / (float(int(n_output))) / (float(len(input_images)))
	activations_PSD = activations_PSD / (float(int(n_output))) / (float(len(input_images)))

	# output mean, variance and PSD
	print('layer %d, mean of activations = %.12f' % (i , activations_mean))
	print('layer %d, variance of activations = %.12f' % (i , activations_variance))
	#print((activations_PSD.shape))

if flag_inference == True:
	top_1_accuracy, top_5_accuracy = run_inference_VGG16(sess , input_string , probabilities)
	print('Summary: top 1 accuracy = %.2f, top 5 accuracy = %.2f.' % (top_1_accuracy, top_5_accuracy))
	
