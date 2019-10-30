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
from alexnet_statistics_convolved_channel_mean import *

import os
import gc

# command line parameters 
# python generate_RD_curves_vgg16.py A B C
# A: which GPU to run
# B: which layer to process
# C: which kernal to process
# e.g., python generate_RD_curves_vgg16.py 3 1 5 -> run GPU3 to do the statistics for the 5th kernal in layer 1.


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

# define quantization hyper-parameters
hist_delta = [0] * num_conv_layers
hist_coded = [0] * num_conv_layers
hist_steps = [0] * num_conv_layers
hist_size = [0] * num_conv_layers
hist_filter_dims = [0] * num_conv_layers
INF = 1000000000


per_channel_means = [0] * num_conv_layers
for i in range(num_conv_layers):
	per_channel_means[i] = np.load('./results_statistics_mean_removal/per_channel_means_%d.npy' % (i))

# define alex model
dropoutPro = 1
classNum = 1000
skip = []
x = tf.placeholder("float", [1, 227, 227, 3]) 
alexnet_model = alexnet_statistics_convolved_channel_mean(x, dropoutPro, classNum, skip, per_channel_means_all = per_channel_means)
output_before_softmax = alexnet_model.fc3
scores = tf.nn.softmax(output_before_softmax)

#conv_inputs = [alexnet_model.CONV_INPUT_1 , alexnet_model.CONV_INPUT_2 , alexnet_model.CONV_INPUT_3 , alexnet_model.CONV_INPUT_4 , alexnet_model.CONV_INPUT_5]
convolved_channel_means = alexnet_model.CONVOLVED_PER_CHANNEL_MEANS


# load alex model
#with tf.Session() as sess:
config = tf.ConfigProto()	
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
alexnet_model.loadModel(sess)


top_1, top_5 = run_inference_alexnet_statistics_convolved_channel_means(sess, x, scores, convolved_channel_means, number_validation_images=num_tesing_images) 

# top_1, top_5 = run_inference_alexnet(sess, x, scores, number_validation_images=num_tesing_images) 

file_results = open("accuracy_alexnet_original.txt", "a+")

file_results.write("%f %f\n" % (top_1 , top_5))

file_results.close()

print('top1 %f top5 %f' % (top_1 , top_5))
	
