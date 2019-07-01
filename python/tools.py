import tensorflow as tf

from tensorflow.contrib import slim
import sys 

SLIM_PATH = './slim/'
sys.path.append(SLIM_PATH)

from nets.vgg import *
from preprocessing import vgg_preprocessing 


def get_all_weights_variables(all_variables):

	variable_weights = []
	
	for i in range(len(all_variables)):

		if "weights" in all_variables[i].name:

			variable_weights.append(all_variables[i])

	return variable_weights


def tensor_preprocessed_input_images(input_string):

	input_images = tf.read_file(input_string)

	input_images = tf.image.decode_jpeg(input_images, channels=3)

	input_images = tf.cast(input_images, tf.float32)

	processed_images = vgg_preprocessing.preprocess_image(input_images, 224, 224, is_training=False)

	processed_images = tf.expand_dims(processed_images, 0)
	
	return processed_images


def read_samples_from_file(filename):

	samples = []

	with open (filename, "r") as f:

		for line in f:

			samples.append(line.strip()) 

		return samples



