clear all;
close all;

neural = alexnet;
layers = neural.Layers;

l = 2;
[h,w,p,q] = size(layers(l).Weights);

% magnitude plot
weights = reshape(permute(layers(2).Weights,[1,2,3,4]),[h*w*p,q]);
vars = squeeze(var(weights));
figure(1);
histogram(weights);

% q-channel-wise
weights = reshape(permute(layers(2).Weights,[1,2,3,4]),[h*w*p,q]);
vars = squeeze(var(weights));
figure(2);
histogram(vars);

% p-channel-wise
weights = reshape(permute(layers(2).Weights,[1,2,4,3]),[h*w*q,p]);
vars = squeeze(var(weights));
figure(3);
histogram(vars);

% q x p globally
weights = reshape(permute(layers(2).Weights,[1,2,3,4]),[h*w,p*q]);
vars = squeeze(var(weights));
figure(4);
histogram(vars);

% layer = quantize(layer);
% net = assembleNetwork(layers);