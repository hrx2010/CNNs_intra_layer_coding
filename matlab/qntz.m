clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/ILSVRC2012_test_00000*.JPEG';
testsize = 128;
maxsteps = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

hist_delta = zeros(maxsteps,q,testsize)*NaN;
hist_coded = zeros(maxsteps,q,testsize)*NaN;
hist_Y_sse = zeros(maxsteps,q,testsize)*NaN;

for i = 1:q % iterate over output channels
    quant = layers;
    convw = quant(l).Weights(:,:,:,i);
    biasw = quant(l).Bias(i);
    scale = 2^floor(log2(sqrt(mean(convw(:).^2))/1024));
    coded = Inf;
    for j = 1:maxsteps
        if coded == 0
            break
        end
        delta = scale*sqrt(2^(j-1));
        % quantize each of the q slices
        convq = quantize(convw,delta);
        biasq = quantize(biasw,delta);
        coded = qentropy([convq(:);biasq(:)]);
        % assemble the net using layers
        quant(l).Weights(:,:,:,i) = convq;
        quant(l).Bias(i) = biasq;
        net = assembleNetwork(quant);
        parfor f = 1:testsize%
            X = imds.readimage(f);
            Y = predict(neural,X);
            Y_ssq = sum(Y(:).^2);
            % run the prediction on image X
            Y_hat = predict(net,X);
            Y_sse = sum((Y_hat(:) - Y(:)).^2);
            hist_delta(j,i,f) = delta;
            hist_coded(j,i,f) = coded;
            hist_Y_sse(j,i,f) = Y_sse;
        
            [~,filename] = fileparts(imds.Files{f});
            disp(sprintf('%s %s | slice %03d, delta: %5.2e, relerr: %5.2e, rate: %5.2e',...
                         archname, filename, i, delta, sqrt(Y_sse/Y_ssq), coded));
        end
    end
end

save(archname,'hist_coded','hist_Y_sse','hist_delta');