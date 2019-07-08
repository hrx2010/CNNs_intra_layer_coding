clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
testsize = 8;
maxsteps = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

hist_base_coded = zeros(maxsteps,q)*NaN;
hist_base_sum_coded = zeros(maxsteps,1,testsize)*NaN;
hist_base_sum_Y_sse = zeros(maxsteps,1,testsize)*NaN;

scale = 2^floor(log2(sqrt(mean(layers(l).Weights(:).^2))/1024));
for j = 1:maxsteps
    delta = scale*sqrt(2^(j-1));
    quant = layers;
    for i = 1:q % iterate over output channels
        convw = quant(l).Weights(:,:,:,i);
        biasw = quant(l).Bias(i);
        % quantize each of the q slices
        convq = quantize(convw,delta);
        biasq = quantize(biasw,0);
        hist_base_coded(j,i) = qentropy([convq(:)]);%;biasq(:)]);
        % assemble the net using layers
        quant(l).Weights(:,:,:,i) = convq;
        quant(l).Bias(i) = biasq;
    end
    net = assembleNetwork(quant);
    for f = 1:testsize%
        X = imds.readimage(f);
        Y = predict(neural,X);
        Y_ssq = sum(Y(:).^2);
        Y_hat = predict(net,X);
        Y_sse = sum((Y_hat(:) - Y(:)).^2);
        hist_base_sum_Y_sse(j,1,f) = Y_sse;
        hist_base_sum_coded(j,1,f) = mean(hist_base_coded(j,:),2);
        [~,filename] = fileparts(imds.Files{f});
        disp(sprintf('%s %s | slice sum, delta: %5.2e, relerr: %5.2e, rate: %5.2e',...
                     archname, filename, delta, sqrt(Y_sse/Y_ssq), hist_base_sum_coded(j,1,f)));
    end
    if hist_base_sum_coded(j,1,1) == 0
        break;
    end
end

%save([archname,'_base_',num2str(testsize)],'hist_base_coded','hist_base_sum_coded','hist_base_sum_Y_sse');