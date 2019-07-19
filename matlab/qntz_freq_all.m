clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012/*.JPEG';
labeldir = '~/Developer/ILSVRC2012/ILSVRC2012_validation_ground_truth.txt';
tranname = 'dft2';
testsize = 1024;
numslope = 64;

[neural,imds] = loadnetwork(archname, imagedir, labeldir);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);
nclass = assembleNetwork(lclass);
trans = {str2func(tranname), str2func(['i',tranname])};

l_inds = findconv(layers); % or specify the layer number directly
l_length = length(l_inds);

load([archname,'_',tranname,'_',num2str(testsize)],'hist_freq_coded','hist_freq_Y_sse','hist_freq_delta');

hist_freq_sum_Y_sse = zeros(numslope,1,testsize)*NaN;
pred_freq_sum_Y_sse = zeros(numslope,1,testsize)*NaN;
hist_freq_sum_coded = zeros(numslope,1,testsize)*NaN;

outputsize = layers(end-1).OutputSize;
Y = zeros(outputsize,testsize);
parfor f = 1:testsize
    X = imds.readimage(f);
    Y(:,f) = predict(neural,X);
end

for j = 1:numslope
    slope = sqrt(2^j)/2^32;
    quant = neural.Layers;
    dists = cell(l_length,1);
    coded = cell(l_length,1);
    steps = cell(l_length,1);
    denom = cell(l_length,1);
    for l = 1:l_length
        l_ind = l_inds(l);
        quant(l_ind).Weights = trans{1}(quant(l_ind).Weights);
        [h,w,p,q] = size(quant(l_ind).Weights);

        hist_freq_coded{l} = hist_freq_coded{l}*(p*q);
        mean_freq_coded = mean(hist_freq_coded{l},3);
        mean_freq_Y_sse = mean(hist_freq_Y_sse{l},3);
        mean_freq_delta = mean(hist_freq_delta{l},3);
        dists{l} = lambda2points(mean_freq_coded,mean_freq_Y_sse,hist_freq_Y_sse{l},slope);
        coded{l} = lambda2points(mean_freq_coded,mean_freq_Y_sse,hist_freq_coded{l},slope);
        steps{l} = lambda2points(mean_freq_coded,mean_freq_Y_sse,hist_freq_delta{l},slope);
        denom{l} = ones(h*w,1)*p*q;

        for i = 1:h*w % iterate over output channels
            [r,c] = ind2sub([h,w],i);
            % quantize for the given lambda
            delta = steps{l}(i,1);
            quant(l_ind).Weights(r,c,:,:) = quantize(quant(l_ind).Weights(r,c,:,:),delta);
        end
        quant(l_ind).Weights = trans{2}(quant(l_ind).Weights);
    end
    neural = assembleNetwork(quant);
    dists = cell2mat(dists);
    coded = cell2mat(coded);
    steps = cell2mat(steps);
    denom = cell2mat(denom);
    parfor f = 1:testsize%
        X = imds.readimage(f);
        Y_hat = predict(neural,X);
        Y_cls = classify(nclass,Y_hat);
        pred_freq_sum_Y_sse(j,1,f) = sum(dists(:,f));
        hist_freq_sum_Y_sse(j,1,f) = mean((Y_hat(:) - Y(:,f)).^2);
        hist_freq_sum_coded(j,1,f) = sum(coded(:,f))/sum(denom);
    end
    disp(sprintf('%s | layer %03d, band sum, lambda: %5.1f, mse: %5.2e (%5.2e), rate: %5.2e',...
         archname, l, log2(slope), mean(hist_freq_sum_Y_sse(j,1,:)), mean(pred_freq_sum_Y_sse(j,1,:)), ...
                 mean(hist_freq_sum_coded(j,1,:))));
    if mean(coded(:)) == 0
        break;
    end
end

save([archname,'_freq_sum_',num2str(testsize)],'hist_freq_sum_coded','hist_freq_sum_Y_sse','pred_freq_sum_Y_sse');