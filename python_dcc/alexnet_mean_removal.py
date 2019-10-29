import tensorflow as tf
import numpy as np
#
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def get_channel_mean(x):
    x_dims = x.get_shape()
    x_height = x_dims[1]
    x_width = x_dims[2]
   
    x_reduced_mean = tf.reduce_mean(x , [0,1,2])
    
    x_repeat_mean = tf.expand_dims(x_reduced_mean , 0)
    x_repeat_mean = tf.expand_dims(x_repeat_mean , 0)
    x_repeat_mean = tf.expand_dims(x_repeat_mean , 0)

    x_repeat_mean = tf.tile(x_repeat_mean , [1 , x_height , x_width , 1])

    return x_repeat_mean

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1 , 
              per_channel_means , convolved_per_channel_means):
    """convolution"""
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("b", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)
		per_channel_means_New = tf.split(value = per_channel_means, num_or_size_splits = groups, axis = 3)
		convolved_per_channel_means_New = tf.split(value = convolved_per_channel_means, num_or_size_splits = groups, axis = 3)
        #featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]

        featureMap = []
        for t1, t2, t3, t4 in zip(xNew , wNew , per_channel_means_New , convolved_per_channel_means_New):
            xNew_mean_removed = t1 - t3
            y = conv(xNew_mean_removed , t2)
            featureMap.append(y + t4)

        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)

class alexnet_mean_removal(object):
    """alexNet model"""
    def __init__(self, x, keepPro, classNum, skip, modelPath = "bvlc_alexnet.npy" , per_channel_means_all , convolved_per_channel_means_all):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        self.buildCNN()
        self.PER_CHANNEL_MEANS_ALL = per_channel_means_all
        self.CONVOLVED_PER_CHANNEL_MEANS_ALL = convolved_per_channel_means_all

    def buildCNN(self):
        """build model"""
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEPPRO)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

    def loadModel(self, sess):
        """load model"""
        wDict = np.load(self.MODELPATH, encoding = "bytes").item()
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        print('p name %s' % name)
                        if len(p.shape) == 1:
                            
                            sess.run(tf.get_variable('b', trainable = False).assign(p))
                        else:
                            sess.run(tf.get_variable('w', trainable = False).assign(p))
