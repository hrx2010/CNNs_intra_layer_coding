import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import slim
import sys 
import cv2
import alexnet

import os

SLIM_PATH = './slim/'
sys.path.append(SLIM_PATH)


ground_truth_file='imagenet_2012_validation_synset_labels_new_index.txt'
batch_size=100
number_validation_images=100
# directory of validation dataset. Please download ImageNet 2010 VAL dataset (totally 50,000 images).
directory_validation_dataset='/home/wangzhe/Documents/data/ImageNet2012/val/'
imgMean = np.array([104, 117, 124], np.float)

num_conv_layers = 5

path_results = './results_statistics_convolved_per_channel_mean' 
if not os.path.exists(path_results):
    os.makedirs(path_results)


def get_channel_mean(x):
    x = tf.constant(x)
    x_dims = x.get_shape()
    x_height = x_dims[1]
    x_width = x_dims[2]
   
    x_reduced_mean = tf.reduce_mean(x , [0,1,2])
    
    x_repeat_mean = tf.expand_dims(x_reduced_mean , 0)
    x_repeat_mean = tf.expand_dims(x_repeat_mean , 0)
    x_repeat_mean = tf.expand_dims(x_repeat_mean , 0)

    x_repeat_mean = tf.tile(x_repeat_mean , [1 , x_height , x_width , 1])

    return x_repeat_mean


def load_ground_truth_from_file(filename):

	with open(filename) as f:
		data = f.readlines()

	ground_truth = [int(x) for x in data]

	return ground_truth


def run_inference_VGG16(sess, input_string, probabilities, number_validation_images=100):

	ground_truth = load_ground_truth_from_file(ground_truth_file)
	
	top_1_accuracy = 0 
	top_5_accuracy = 0 

	for id_batch in range(int(np.ceil(number_validation_images/np.float(batch_size)))):

		id_start_image = id_batch * batch_size + 1
		id_end_image = np.minimum(number_validation_images + 1, id_start_image+batch_size)

		for i in range(id_start_image, id_end_image, 1): 

			img_path = directory_validation_dataset + 'ILSVRC2012_val_%08d.JPEG' % (i)

			prediction_score = sess.run(probabilities, feed_dict={input_string:img_path})

			labels_ranked = np.argsort(prediction_score[0][:])[::-1]

			if labels_ranked[0] == ground_truth[i-1]:
				top_1_accuracy += 1

			for j in range(5):
				if labels_ranked[j] == ground_truth[i-1]:
					top_5_accuracy += 1

		print('Process inference on [%d / %d] validation images at ImageNet.' % (id_end_image , number_validation_images))

	return 100.0*top_1_accuracy/np.float(id_end_image-1), 100.0*top_5_accuracy/np.float(id_end_image-1)


def run_inference_alexnet(sess, input_img, probabilities, number_validation_images=100):

	ground_truth = load_ground_truth_from_file(ground_truth_file)
	
	top_1_accuracy = 0 
	top_5_accuracy = 0 

	last_layer_outputs = [0] * number_validation_images

	for id_batch in range(int(np.ceil(number_validation_images/np.float(batch_size)))):

		id_start_image = id_batch * batch_size + 1
		id_end_image = np.minimum(number_validation_images + 1, id_start_image+batch_size)

		for i in range(id_start_image, id_end_image, 1): 

			img_path = directory_validation_dataset + 'ILSVRC2012_val_%08d.JPEG' % (i)
			img = cv2.imread(img_path)
			img_resized = cv2.resize(img.astype(float), (227, 227))
			img_resized -= imgMean
			img_resized = img_resized.reshape((1, 227, 227, 3))

			prediction_score = sess.run(probabilities, feed_dict={input_img:img_resized})

			labels_ranked = np.argsort(prediction_score[0][:])[::-1]

			if labels_ranked[0] == ground_truth[i-1]:
				top_1_accuracy += 1

			for j in range(5):
				if labels_ranked[j] == ground_truth[i-1]:
					top_5_accuracy += 1

		print('Process inference on [%d / %d] validation images at ImageNet.' % (id_end_image , number_validation_images))

	return 100.0*top_1_accuracy/np.float(id_end_image-1), 100.0*top_5_accuracy/np.float(id_end_image-1)


def run_inference_alexnet_statistics(sess, input_img, probabilities, conv_inputs, conv_outputs, number_validation_images=100):

	ground_truth = load_ground_truth_from_file(ground_truth_file)
	
	top_1_accuracy = 0 
	top_5_accuracy = 0 

	last_layer_outputs = [0] * number_validation_images

	per_channel_means = [0] * num_conv_layers
	convolved_per_channel_means = [0] * num_conv_layers

	for id_batch in range(int(np.ceil(number_validation_images/np.float(batch_size)))):

		id_start_image = id_batch * batch_size + 1
		id_end_image = np.minimum(number_validation_images + 1, id_start_image+batch_size)

		for i in range(id_start_image, id_end_image, 1): 

			img_path = directory_validation_dataset + 'ILSVRC2012_val_%08d.JPEG' % (i)
			img = cv2.imread(img_path)
			img_resized = cv2.resize(img.astype(float), (227, 227))
			img_resized -= imgMean
			img_resized = img_resized.reshape((1, 227, 227, 3))

			prediction_score = sess.run(probabilities, feed_dict={input_img:img_resized})

			labels_ranked = np.argsort(prediction_score[0][:])[::-1]

			if labels_ranked[0] == ground_truth[i-1]:
				top_1_accuracy += 1

			for j in range(5):
				if labels_ranked[j] == ground_truth[i-1]:
					top_5_accuracy += 1

			for k in range(num_conv_layers):
				conv_input = sess.run(conv_inputs[k], feed_dict={input_img:img_resized})
				conv_output = sess.run(conv_outputs[k], feed_dict={input_img:img_resized})


				if i == 1:
					per_channel_means[k]  = conv_input
					convolved_per_channel_means[k] = conv_output
					#print(conv_input)
				else:
					per_channel_means[k] = per_channel_means[k] + conv_input
					convolved_per_channel_means[k] = convolved_per_channel_means[k] + conv_output

		print('Process inference on [%d / %d] validation images at ImageNet.' % (id_end_image , number_validation_images))

	for k in range(num_conv_layers):
		per_channel_means[k] /= (1.0 * number_validation_images)
		# convolved_per_channel_means[k] /= (1.0 * number_validation_images)

		per_channel_means[k] = get_channel_mean(per_channel_means[k])

		results = sess.run(per_channel_means[k])
		
		filename_1 = ('%s/per_channel_means_%d.npy' % (path_results , k))
		#filename_2 = ('%s/convolved_per_channel_means_%d.npy' % (path_results , k))
		np.save(filename_1 , results)
		#np.save(filename_2 , convolved_per_channel_means[k])
		

	return 100.0*top_1_accuracy/np.float(id_end_image-1), 100.0*top_5_accuracy/np.float(id_end_image-1)



def run_inference_alexnet_statistics_convolved_channel_means(sess, input_img, probabilities, convolved_channel_means, number_validation_images=100):

	ground_truth = load_ground_truth_from_file(ground_truth_file)
	
	top_1_accuracy = 0 
	top_5_accuracy = 0 

	last_layer_outputs = [0] * number_validation_images


	for id_batch in range(int(np.ceil(number_validation_images/np.float(batch_size)))):

		id_start_image = id_batch * batch_size + 1
		id_end_image = np.minimum(number_validation_images + 1, id_start_image+batch_size)

		for i in range(id_start_image, id_end_image, 1): 

			img_path = directory_validation_dataset + 'ILSVRC2012_val_%08d.JPEG' % (i)
			img = cv2.imread(img_path)
			img_resized = cv2.resize(img.astype(float), (227, 227))
			img_resized -= imgMean
			img_resized = img_resized.reshape((1, 227, 227, 3))

			prediction_score = sess.run(probabilities, feed_dict={input_img:img_resized})

			labels_ranked = np.argsort(prediction_score[0][:])[::-1]

			if labels_ranked[0] == ground_truth[i-1]:
				top_1_accuracy += 1

			for j in range(5):
				if labels_ranked[j] == ground_truth[i-1]:
					top_5_accuracy += 1

			for k in range(num_conv_layers):
				values = sess.run(convolved_channel_means[k], feed_dict={input_img:img_resized})
				filename_1 = ('%s/convolved_per_channel_means_%d.npy' % (path_results , k))
				np.save(filename_1 , values)

		print('Process inference on [%d / %d] validation images at ImageNet.' % (id_end_image , number_validation_images))
		

	return 100.0*top_1_accuracy/np.float(id_end_image-1), 100.0*top_5_accuracy/np.float(id_end_image-1)



def run_inference_VGG16_mini_batch(sess, input_string, probabilities, number_validation_images=100):
	ground_truth = load_ground_truth_from_file(ground_truth_file)
	
	top_1_accuracy = 0 
	top_5_accuracy = 0 

	for id_batch in range(int(np.ceil(number_validation_images/np.float(batch_size)))):
		id_start_image = id_batch * batch_size + 1
		id_end_image = np.minimum(number_validation_images + 1, id_start_image+batch_size)

		imgs_path = []

		for i in range(id_start_image, id_end_image, 1): 
			img_path = directory_validation_dataset + 'ILSVRC2012_val_%08d.JPEG' % (i)
			imgs_path.append(img_path)

		prediction_score = sess.run(probabilities, feed_dict={input_string:imgs_path})

		for i in range(id_start_image, id_end_image, 1):
			labels_ranked = np.argsort(prediction_score[i][:])[::-1]

			if labels_ranked[0] == ground_truth[i-1]:
				top_1_accuracy += 1

			for j in range(5):
				if labels_ranked[j] == ground_truth[i-1]:
					top_5_accuracy += 1

		print('Process inference on [%d / %d] validation images at ImageNet.' % (id_end_image , number_validation_images))

	return 100*top_1_accuracy/np.float(id_end_image-1), 100*top_5_accuracy/np.float(id_end_image-1)


def run_inference_VGG16_last_layer_output(sess, input_string, logits, number_validation_images=100):

	ground_truth = load_ground_truth_from_file(ground_truth_file)
	
	top_1_accuracy = 0 
	top_5_accuracy = 0 

	last_layer_outputs = [0] * number_validation_images

	for id_batch in range(int(np.ceil(number_validation_images/np.float(batch_size)))):

		id_start_image = id_batch * batch_size + 1
		id_end_image = np.minimum(number_validation_images + 1, id_start_image+batch_size)

		for i in range(id_start_image, id_end_image, 1): 

			img_path = directory_validation_dataset + 'ILSVRC2012_val_%08d.JPEG' % (i)

			
			output = sess.run(logits, feed_dict={input_string:img_path})

			last_layer_outputs[i - 1] = output

	return last_layer_outputs


def run_inference_alexnet_last_layer_output(sess, input_img, logits, number_validation_images=100):

	ground_truth = load_ground_truth_from_file(ground_truth_file)
	
	top_1_accuracy = 0 
	top_5_accuracy = 0 

	last_layer_outputs = [0] * number_validation_images

	for id_batch in range(int(np.ceil(number_validation_images/np.float(batch_size)))):

		id_start_image = id_batch * batch_size + 1
		id_end_image = np.minimum(number_validation_images + 1, id_start_image+batch_size)

		for i in range(id_start_image, id_end_image, 1): 

			img_path = directory_validation_dataset + 'ILSVRC2012_val_%08d.JPEG' % (i)
			img = cv2.imread(img_path)
			img_resized = cv2.resize(img.astype(float), (227, 227))
			img_resized -= imgMean
			img_resized = img_resized.reshape((1, 227, 227, 3))

			output = sess.run(logits, feed_dict={input_img: img_resized})

			last_layer_outputs[i - 1] = output

	return last_layer_outputs


def run_inference_VGG16_last_layer_output_mini_batch(sess, input_string, logits, number_validation_images=100):
	ground_truth = load_ground_truth_from_file(ground_truth_file)
	top_1_accuracy = 0 
	top_5_accuracy = 0 

	last_layer_outputs = [0] * number_validation_images

	for id_batch in range(int(np.ceil(number_validation_images/np.float(batch_size)))):
		id_start_image = id_batch * batch_size + 1
		id_end_image = np.minimum(number_validation_images + 1, id_start_image+batch_size)

		imgs_path = []

		for i in range(id_start_image, id_end_image, 1): 
			img_path = directory_validation_dataset + 'ILSVRC2012_val_%08d.JPEG' % (i)
			imgs_path.append(img_path)

		output = sess.run(logits, feed_dict={input_string:imgs_path})
		for i in range(id_start_image, id_end_image, 1):			
			last_layer_outputs[i - 1] = output[i - 1]

	return last_layer_outputs
