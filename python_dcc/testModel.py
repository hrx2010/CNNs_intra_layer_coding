
import os
import urllib.request
import argparse
import sys
import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes
import glob

from tensorflow.contrib import slim 

dropoutPro = 1
classNum = 1000
skip = []
#get testImage
testPath = "testModel"
testImg = []

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*')) # so there is no problem with hidden files

for f in listdir_nohidden(testPath):
    #print(f)
    testImg.append(cv2.imread(f))
 
imgMean = np.array([104, 117, 124], np.float)
x = tf.placeholder("float", [1, 227, 227, 3])
 
model = alexnet.alexNet(x, dropoutPro, classNum, skip)
score = model.fc3
softmax = tf.nn.softmax(score)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.loadModel(sess) #Load the model
 
    #variables_to_restore = slim.get_variables_to_restore()
    #for i in len(variables_to_restore):
        #print(variables_to_restore[i].name)

    with tf.variable_scope('conv1', reuse = True):
        node = tf.get_variable('b', trainable = False)
        print(sess.run(node))

    # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv2'):
        # print(i)

    for i, img in enumerate(testImg):
        #img preprocess
        test = cv2.resize(img.astype(float), (227, 227)) #resize
        test -= imgMean #subtract image mean
        test = test.reshape((1, 227, 227, 3)) #reshape into tensor shape
        maxx = sess.run(softmax, feed_dict = {x: test})
        #res = caffe_classes.class_names[maxx] #find the max probility
        print(maxx)

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 0, 255), 2) #putting on the labels
        #cv2.imshow("demo", img) 
        #cv2.waitKey(0)

