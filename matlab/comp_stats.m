clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
tranname = 'dft2';
testsize = 1024;
maxsteps = 64;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);
nclass = assembleNetwork(lclass);
trans = {str2func(tranname), str2func(['i',tranname])};

l_kernel = findconv(neural.Layers);
l_length = length(l_kernel);

hist_delta = cell(l_length,1);
hist_coded = cell(l_length,1);
hist_W_sse = cell(l_length,1);
hist_Y_sse = cell(l_length,1);
hist_Y_top = cell(l_length,1);

hist_gainX = cell(l_length,1);
hist_gainT = cell(l_length,1);

layers = neural.Layers(l_kernel);
for l = 1:l_length
    X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
    [H,W,P,Q] = size(X);
    [h,w,p,q,g] = size(layers(l).Weights);
    T = reshape(permute(layers(l).Weights,[1,2,3,5,4]),[h,w,p*g,q]);

    X_psd = real(diag3(conj(permute(blkfft2(conj(permute(blkfft2(blktoeplitz(autocorr2(X,h,w)),h,w),[2,1,3])),h,w),[2,1,3]))));
    T_psd = mean(abs(fft2(T)).^2,4)*(1/h/w);

    hist_gainX{l} = mean(X_psd(:))/geomean(X_psd(:));
    hist_gainT{l} = mean(T_psd(:))/geomean(T_psd(:));

    disp(sprintf('Coding gain for layer %03d is %5.2f (%5.2f, %5.2f) dB (%5d coefficients)', ...
                 l, 10*log10(mean(hist_gainX{l}.*hist_gainT{l})), 10*log10(mean(hist_gainX{l})), ...
                 10*log10(mean(hist_gainT{l})), h*w*p*q*g));
end