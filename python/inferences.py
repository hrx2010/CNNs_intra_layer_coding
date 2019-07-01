import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import slim
import sys 

SLIM_PATH = './slim/'
sys.path.append(SLIM_PATH)


ground_truth_file='imagenet_2012_validation_synset_labels_new_index.txt'
batch_size=100
number_validation_images=1000
# directory of validation dataset. Please download ImageNet 2010 VAL dataset (totally 50,000 images).
directory_validation_dataset='/home/wangzhe/Documents/data/ImageNet2012/val/'


def load_ground_truth_from_file(filename):

	with open(filename) as f:
		data = f.readlines()

	ground_truth = [int(x) for x in data]

	return ground_truth


def run_inference_VGG16(sess, input_string, probabilities):

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

	return 100*top_1_accuracy/np.float(id_end_image-1), 100*top_5_accuracy/np.float(id_end_image-1)
