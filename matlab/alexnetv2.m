function net = alexnetv2
    load('ilsvrc_classes.mat','classes');
    layers = [imageInputLayer([224,224,3],'Name','data'),

              convolution2dLayer([11,11],64,'Stride',4,'Padding',2,'Name','conv1'),
              reluLayer('Name','relu1'),
              maxPooling2dLayer([3,3],'Stride',2,'Padding',0,'Name','pool1'),

              convolution2dLayer([5,5],192,'Stride',1,'Padding',2,'Name','conv2'),
              reluLayer('Name','relu2'),
              maxPooling2dLayer([3,3],'Stride',2,'Padding',0,'Name','pool2'),

              convolution2dLayer([3,3],384,'Stride',1,'Padding',1,'Name','conv3'),
              reluLayer('Name','relu3'),
              convolution2dLayer([3,3],256,'Stride',1,'Padding',1,'Name','conv4'),
              reluLayer('Name','relu4'),
              convolution2dLayer([3,3],256,'Stride',1,'Padding',1,'Name','conv5'),
              reluLayer('Name','relu5'),
              maxPooling2dLayer([3,3],'Stride',2,'Padding',0,'Name','pool5'),
              
              dropoutLayer('Name','drop6'),
              fullyConnectedLayer(4096,'Name','fc6'),
              reluLayer('Name','relu6'),
              
              dropoutLayer('Name','drop7'),
              fullyConnectedLayer(4096,'Name','fc7'),
              reluLayer('Name','relu7'),

              fullyConnectedLayer(1000,'Name','fc8'),
              softmaxLayer('Name','prob'),
              classificationLayer('Name','output','Classes',classes)
             ];

    net = SeriesNetwork(layers);
end
