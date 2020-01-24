clear all;
close all;

[~,layers] = seannet;

[XTrain, YTrain] = digitTrain4DArrayData;

img = imread('~/Desktop/Stanford_GSB_Logo.jpg');
img = single(img(:,17:340,1)>210 & img(:,17:340,2)>210 & img(:,17:340,3)>210);
YTrain = reshape(reshape(categorical(img),1,1,1,[]),[],1);

[X1, X2] = meshgrid(1:size(img,2),1:size(img,1));
XTrain = reshape([X1(:)';X2(:)'],2,1,1,[]);

options = trainingOptions('adam','Plots','training-progress');
neural = trainNetwork(XTrain,YTrain,layers,options);
