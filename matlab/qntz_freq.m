clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
tranname = 'dft2';
testsize = 1024;
maxsteps = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);
trans = {str2func(tranname), str2func(['i',tranname])};

l_inds = findconv(layers); % or specify the layer number directly
l_length = length(l_inds);

hist_freq_delta = cell(l_length,1);
hist_freq_coded = cell(l_length,1);
hist_freq_Y_sse = cell(l_length,1);

outputsize = layers(end-1).OutputSize;
Y = zeros(outputsize,testsize);
parfor f = 1:testsize
    X = imds.readimage(f);
    Y(:,f) = predict(neural,X);
end

for l = 1:l_length
    layers = neural.Layers;
    l_ind = l_inds(l);
    layers(l_ind).Weights = trans{1}(layers(l_ind).Weights);
    [h,w,p,q] = size(layers(l_ind).Weights);

    hist_delta = zeros(maxsteps,h*w,testsize)*NaN;
    hist_coded = zeros(maxsteps,h*w,testsize)*NaN;
    hist_Y_sse = zeros(maxsteps,h*w,testsize)*NaN;
    
    for i = 1:h*w % iterate over the frequency bands
        [r,c] = ind2sub([h,w],i);
        scale = 2^floor(log2(sqrt(mean(reshape(layers(l_ind).Weights(r,c,:,:),[],1).^2))/1024));
        coded = Inf;
        for j = 1:maxsteps
            delta = scale*sqrt(2^(j-1));
            % quantize each of the q slices
            quant = layers;
            quant(l_ind).Weights(r,c,:,:) = quantize(quant(l_ind).Weights(r,c,:,:),delta);
            coded = qentropy(quant(l_ind).Weights(r,c,:,:));
            % assemble the net using layers
            quant(l_ind).Weights = trans{2}(quant(l_ind).Weights);
            neural = assembleNetwork(quant);
            parfor f = 1:testsize%
                X = imds.readimage(f);
                % run the prediction on image X
                Y_hat = predict(neural,X);
                Y_sse = mean((Y_hat(:) - Y(:,f)).^2);
                hist_delta(j,i,f) = delta;
                hist_coded(j,i,f) = coded;
                hist_Y_sse(j,i,f) = Y_sse;
            end
            disp(sprintf('%s %s | layer %03d, band %03d, scale: %3d, delta: %+5.1f, mse: %5.2e, rate: %5.2e',...
                         archname, tranname, l, i, log2(scale), log2(delta), mean(hist_Y_sse(j,i,:)), coded));
            if coded == 0
                break
            end
        end
    end
    
    hist_freq_delta{l} = hist_delta;
    hist_freq_coded{l} = hist_coded;
    hist_freq_Y_sse{l} = hist_Y_sse;
end
save([archname,'_',tranname,'_',num2str(testsize)],'hist_freq_coded','hist_freq_Y_sse','hist_freq_delta');