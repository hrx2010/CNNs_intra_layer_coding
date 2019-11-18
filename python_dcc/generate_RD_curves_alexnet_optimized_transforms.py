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
from alexnet_mean_removal import * 

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

# transform method
transform_method = str(sys.argv[4])

# transform data path
transform_data_path = str(sys.argv[5])

# flag mean removal
flag_mean_removal = str(sys.argv[6])

transform_matrices = load_transform_data_from_file(transform_data_path)

def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))


#create folder of results
path_results = ('./results_RD_curves_alexnet_%s' % (transform_method))
if flag_mean_removal == 'm':
	path_results = path_results + '_mean_removal'

if not os.path.exists(path_results):
    os.makedirs(path_results)


# the directory of sample images.
SAMPLES_PATH = './samples/'

 
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


per_channel_means = [0] * num_conv_layers
convolved_per_channel_means = [0] * num_conv_layers


for i in range(num_conv_layers):
	per_channel_means[i] = np.load('./results_statistics_mean_removal/per_channel_means_%d.npy' % (i))
	convolved_per_channel_means[i] = np.load('./results_statistics_convolved_per_channel_mean/convolved_per_channel_means_%d.npy' % (i))


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


if flag_mean_removal == 'm':
	alexnet_model = alexnet_mean_removal(x, dropoutPro, classNum, skip, per_channel_means_all = per_channel_means, convolved_per_channel_means_all = convolved_per_channel_means)
else:
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
	if i != processing_layer_id:
		continue

	with tf.variable_scope(conv_names[i] , reuse = True):
		node_weights = tf.get_variable('w', trainable = False)

	weight_values_original = sess.run(node_weights)

	[fh, fw, n_input, n_output] = weight_values_original.shape

	hist_delta[i] = np.zeros((fh * fw , max_rates))
	hist_coded[i] = np.zeros((fh * fw , max_rates))
	hist_steps[i] = np.zeros((fh * fw , max_rates))
	hist_size[i] = np.zeros((fh * fw , max_rates))

	weight_values_original_transformed = transform_given_matrices(transform_matrices[i][0] , weight_values_original)

	weight_values_original_transformed = np.array(weight_values_original_transformed)

	[dim1 , dim2] = transform_matrices[i][0].shape
	inv_transform_matrices_reshaped = np.zeros((n_input * n_output , fh * fw , fh * fw))

	cnt = 0	

	if dim2 == 1:

		for p in range(n_output):
			for q in range(n_input):
				trans_mat = transform_matrices[i][0][q][0]
				inv_trans_mat = np.linalg.inv(trans_mat)
				inv_transform_matrices_reshaped[cnt , : , :] = inv_trans_mat
				cnt = cnt + 1

	elif dim2 == 2:
		half_n_output = int(n_output / 2)

		for p in range(half_n_output):
			for q in range(n_input):
				trans_mat = transform_matrices[i][0][q][0]
				inv_trans_mat = np.linalg.inv(trans_mat)
				inv_transform_matrices_reshaped[cnt , : , :] = inv_trans_mat
				cnt = cnt + 1

		for p in range(half_n_output):
			for q in range(n_input):
				trans_mat = transform_matrices[i][0][q][1]
				inv_trans_mat = np.linalg.inv(trans_mat)
				inv_transform_matrices_reshaped[cnt , : , :] = inv_trans_mat
				cnt = cnt + 1


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
				
				quantized_weights = np.array(weight_values_original_transformed)
				quantized_weights[r,c,:,:] = quantize(quantized_weights[r,c,:,:] , np.power(2 , delta) , B) #B
				coded = fixed_length_entropy(quantized_weights[r,c,:,:] , B) * n_input * n_output #B

				
				inverted_quantized_weights = np.real(i_transform_given_matrices_fast(inv_transform_matrices_reshaped , quantized_weights , sess))

				sess.run(node_weights.assign(inverted_quantized_weights))

				
				last_layer_output_quantized = run_inference_alexnet_last_layer_output(sess, x, output_before_softmax)
				last_layer_output_quantized = np.array(last_layer_output_quantized)

				cur_output_error = np.mean((last_layer_output - last_layer_output_quantized)**2)
				cur_mse = 0 

				if t == 0 or cur_output_error < min_output_error:
					min_output_error = cur_output_error
					hist_delta[i][j][k] = cur_output_error
					hist_coded[i][j][k] = coded
					hist_steps[i][j][k] = t
					hist_size[i][j][k] = delta

				if t > 0 and last_output_error <= cur_output_error and last_mse <= cur_mse:
					offset = delta - 2
					break

				print('layer %d , kernal (%d , %d) [%d , %d], rate %d bits, step %d, delta %f, total rate %f, output error %.14f (min error %.14f)' % (i , r , c , fh , fw , k , t , delta , coded , cur_output_error , hist_delta[i][j][k]))

				if k == 0:
					break
				
				last_output_error = cur_output_error
				last_mse = cur_mse
			
			file_results.write("%d %d %d %f %.14f %d %.10f\n" % (i , j , k , hist_coded[i][j][k] , hist_delta[i][j][k] , hist_steps[i][j][k] , hist_size[i][j][k]))

			gc.collect()

file_results.close()
	
