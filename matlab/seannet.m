function [neural,layers] = seannet
    layers = [imageInputLayer([2,1,1],'Name','data','Normalization','none'),

              fullyConnectedLayer(2*8,'Name','fc1'),
              reluLayer('Name','relu1'),

              fullyConnectedLayer(2*16,'Name','fc2'),
              reluLayer('Name','relu2'),

              fullyConnectedLayer(2*32,'Name','fc3'),
              reluLayer('Name','relu3'),

              fullyConnectedLayer(2*16,'Name','fc4'),
              reluLayer('Name','relu4'),
              
              fullyConnectedLayer(2*8,'Name','fc5'),
              reluLayer('Name','relu5'),
              
              fullyConnectedLayer(2*1,'Name','fc6'),
              softmaxLayer('Name','tanh'),
              classificationLayer('Name','output')
             ];

    neural = SeriesNetwork(layers);
end
