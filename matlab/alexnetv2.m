function net = alexnetv2
    load('ilsvrc_classes.mat','classes');
    layers = [imageInputLayer([224,224,3],'Name','data'),

              convolution2dLayer([11,11],64,'Stride',4,'Padding',2,'Name','conv1','Weights',zeros(11,11,3,64),'Bias',zeros(1,1,64)),
              reluLayer('Name','relu1'),
              maxPooling2dLayer([3,3],'Stride',2,'Padding',0,'Name','pool1'),

              convolution2dLayer([5,5],192,'Stride',1,'Padding',2,'Name','conv2','Weights',zeros(5,5,64,192),'Bias',zeros(1,1,192)),
              reluLayer('Name','relu2'),
              maxPooling2dLayer([3,3],'Stride',2,'Padding',0,'Name','pool2'),

              convolution2dLayer([3,3],384,'Stride',1,'Padding',1,'Name','conv3','Weights',zeros(3,3,192,384),'Bias',zeros(1,1,384)),
              reluLayer('Name','relu3'),
              convolution2dLayer([3,3],256,'Stride',1,'Padding',1,'Name','conv4','Weights',zeros(3,3,384,256),'Bias',zeros(1,1,256)),
              reluLayer('Name','relu4'),
              convolution2dLayer([3,3],256,'Stride',1,'Padding',1,'Name','conv5','Weights',zeros(3,3,256,256),'Bias',zeros(1,1,256)),
              reluLayer('Name','relu5'),
              maxPooling2dLayer([3,3],'Stride',2,'Padding',0,'Name','pool5'),
              
              dropoutLayer('Name','drop6'),
              convolution2dLayer([6,6],4096,'Name','fc6','Weights',zeros(6,6,256,4096),'Bias',zeros(1,1,4096)),
              reluLayer('Name','relu6'),
              
              dropoutLayer('Name','drop7'),
              convolution2dLayer([1,1],4096,'Name','fc7','Weights',zeros(1,1,4096,4096),'Bias',zeros(1,1,4096)),
              reluLayer('Name','relu7'),

              convolution2dLayer([1,1],1000,'Name','fc8','Weights',zeros(1,1,4096,1000),'Bias',zeros(1,1,1000)),
              softmaxLayer('Name','prob'),
              classificationLayer('Name','output','Classes',classes)
             ];

    net = SeriesNetwork(layers);
end
