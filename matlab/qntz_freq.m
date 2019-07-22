clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'densenet201';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
tranname = 'dft2';
testsize = 1024;
maxsteps = 64;

[neural,imds] = loadnetwork(archname, imagedir, labeldir);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);
nclass = assembleNetwork(lclass);
trans = {str2func(tranname), str2func(['i',tranname])};

l_inds = findconv(layers); % or specify the layer number directly
l_length = length(l_inds);

hist_delta = cell(l_length,1);
hist_coded = cell(l_length,1);
hist_W_sse = cell(l_length,1);
hist_Y_sse = cell(l_length,1);
hist_Y_top = cell(l_length,1);

outputsize = layers(end-1).OutputSize;
Y = zeros(outputsize,testsize);
parfor f = 1:testsize
    X = imds.readimage(f);
    Y(:,f) = predict(neural,X);
end

for l = 1:l_length
    l_ind = l_inds(l);
    layer = neural.Layers(l_ind);
    layer.Weights = trans{1}(layer.Weights);
    % layers = neural.Layers;
    % layers(l_ind).Weights = trans{1}(layers(l_ind).Weights);
    [h,w,p,q] = size(layer.Weights);

    deltas = zeros(maxsteps,h*w,1)*NaN;
    codeds = zeros(maxsteps,h*w,1)*NaN;
    W_sses = zeros(maxsteps,h*w,1)*NaN;
    Y_sses = zeros(maxsteps,h*w,testsize)*NaN;
    Y_tops = zeros(maxsteps,h*w,testsize)*NaN;
    
    for i = 1:h*w % iterate over the frequency bands
        [r,c] = ind2sub([h,w],i);
        scale = 2^floor(log2(sqrt(mean(reshape(layer.Weights(r,c,:,:),[],1).^2))/1024));
        coded = Inf;
        for j = 1:maxsteps
            % quantize each of the q slices
            quant = layer;
            delta = scale*sqrt(2^(j-1));
            quant.Weights(r,c,:,:) = quantize(quant.Weights(r,c,:,:),delta);
            coded = qentropy(quant.Weights(r,c,:,:))*(p*q);
            % assemble the net using layers
            quant.Weights = trans{2}(quant.Weights);
            W_sse = sum(reshape(quant.Weights(r,c,:,:) - neural.Layers(l_ind).Weights(r,c,:,:),[],1).^2);
            ournet = modifyLayers(neural,quant);

            parfor f = 1:testsize
                X = imds.readimage(f);
                % run the prediction on image X
                Y_hat = predict(ournet,X);
                Y_cls = classify(nclass,Y_hat(:));
                Y_tops(j,i,f) = Y_cls == imds.Labels(f);
                Y_sses(j,i,f) = mean((Y_hat(:) - Y(:,f)).^2);
            end
            deltas(j,i,1) = delta;
            codeds(j,i,1) = coded;
            W_sses(j,i,1) = W_sse;

            disp(sprintf('%s %s | layer: %03d/%03d, band: %03d/%03d, scale: %3d, delta: %+5.1f, ymse: %5.2e, wmse: %5.2e, top1: %4.1f, rate: %5.2e', ...
                 archname, tranname, l, l_length, i, h*w, log2(scale), log2(delta), mean(Y_sses(j,i,:)), W_sse, ...
                 mean(Y_tops(j,i,:))*100, coded/(p*q)));
            if coded == 0
                break
            end
        end
    end
    hist_delta{l} = deltas;
    hist_coded{l} = codeds;
    hist_Y_top{l} = Y_tops;
    hist_Y_sse{l} = Y_sses;
    hist_W_sse{l} = W_sses;
end
save([archname,'_',tranname,'_val_',num2str(testsize)],'hist_coded','hist_Y_sse','hist_Y_top','hist_delta','hist_W_sse');