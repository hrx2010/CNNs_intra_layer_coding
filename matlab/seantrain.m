clear all;
close all;

[~,layers] = seannet;

%[XTrain, YTrain] = digitTrain4DArrayData;

img = imread('./stanford.bmp');
img = (img(:,:,1)>210 & img(:,:,2)>210 & img(:,:,3)>210);
Y = reshape(reshape(categorical(img),1,1,1,[]),[],1);

[X1, X2] = meshgrid(1:size(img,2),1:size(img,1));
X = reshape([X1(:)';X2(:)'],2,1,1,[]);

options = trainingOptions('adam','Plots','training-progress','MaxEpochs',48,...
                          'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,'LearnRateDropPeriod',16);
neural = trainNetwork(X,img,layers,options);
