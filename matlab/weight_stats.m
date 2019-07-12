clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
testsize = 1;
maxsteps = 1;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);
m = 1;  %row
n = 1;  %col

% IM IM RE
% IM DC RE
% IM RE RE

xcoeffs = fft2split(fftshift(fftshift(fft2(neural.Layers(l).Weights),1),2));
weights = neural.Layers(l).Weights;

h = size(xcoeffs,1);
w = size(xcoeffs,2);

x = xcoeffs(:);
%x = neural.Layers(l).Weights(:);

bin_width = 2^-4;
bin_count = 1024;
bin_bound = 
bin_edges = (-bin_bound+0.5:bin_bound-0.5)*bin_width;
bin_point = (-bin_bound+1.0:bin_bound-1.0)*bin_width;

x(x<bin_edges(1)) = bin_edges(1);
x(x>bin_edges(end)) = bin_edges(end);

histcount = histcounts(x,bin_edges);
plot(bin_point,histcount);
xlabel(sprintf('Layer %d. Coeff. %d,%d.', l, m, n));
ylabel('Frequency');
%histogram(x,bin_edges','Normalization','pdf');
disp(sprintf('kurtosis: %f', kurtosis(x)));
disp(sprintf('skewness: %f', skewness(x)));
