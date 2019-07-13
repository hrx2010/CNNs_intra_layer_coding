clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
testsize = 1024;
maxsteps = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

hist_freq_delta = zeros(maxsteps,h*w,testsize)*NaN;
hist_freq_coded = zeros(maxsteps,h*w,testsize)*NaN;
hist_freq_Y_sse = zeros(maxsteps,h*w,testsize)*NaN;

layers(l).Weights = dft2(layers(l).Weights);

for i = 1:h*w % iterate over the frequency bands
    [r,c] = ind2sub([h,w],i);
    scale = 2^floor(log2(sqrt(mean(reshape(layers(l).Weights(r,c,:,:),[],1).^2))/1024));
    coded = Inf;
    for j = 1:maxsteps
        delta = scale*sqrt(2^(j-1));
        % quantize each of the q slices
        quant = layers;
        quant(l).Weights(r,c,:,:) = quantize(quant(l).Weights(r,c,:,:),delta);
        coded = qentropy(quant(l).Weights(r,c,:,:));
        % assemble the net using layers
        quant(l).Weights = idft2(quant(l).Weights);
        net = assembleNetwork(quant);
        for f = 1:testsize%
            X = imds.readimage(f);
            Y = predict(neural,X);
            Y_ssq = sum(Y(:).^2);
            % run the prediction on image X
            Y_hat = predict(net,X);
            Y_sse = sum((Y_hat(:) - Y(:)).^2);
            hist_freq_delta(j,i,f) = delta;
            hist_freq_coded(j,i,f) = coded;
            hist_freq_Y_sse(j,i,f) = Y_sse;
        
            [~,filename] = fileparts(imds.Files{f});
            disp(sprintf('%s %s | band %03d, scale: %3d, delta: %+5.1f, relerr: %5.2e, rate: %5.2e',...
                         archname, filename, i, log2(scale), log2(delta), sqrt(Y_sse/Y_ssq), coded));
        end
        if coded == 0
            break
        end
    end
end

save([archname,'_freq_',num2str(testsize)],'hist_freq_coded','hist_freq_Y_sse','hist_freq_delta');