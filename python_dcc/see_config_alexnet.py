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
from alexnet import *

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

# define transform method
transform_method = str(sys.argv[4])

	
def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))


#create folder of results
path_results = ('./results_RD_curves_alexnet_%s' % (transform_method)) 
if not os.path.exists(path_results):
    os.makedirs(path_results)


# the directory of sample images.
SAMPLES_PATH = './samples/' 
# model name.
model='alexnet_mean_removal' 
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
# transform_method = 'dft2'

print('asdasdasd %d' % (len(conv_names)))

# define quantization hyper-parameters
max_steps = 50
max_rates = 17
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


for i in range(num_conv_layers):
	with tf.variable_scope(conv_names[i] , reuse = True):
		node_weights = tf.get_variable('w', trainable = False)

	weight_values_original = sess.run(node_weights)	

	[fh, fw, n_input, n_output] = weight_values_original.shape


	print('%d: %d %d %d %d' % (i, fh, fw, n_input, n_output))
