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

l_inds = findconv(neural.Layers); % or specify the layer number directly
l_length = length(l_inds);

hist_delta = cell(l_length,1);
hist_coded = cell(l_length,1);
hist_W_sse = cell(l_length,1);
hist_Y_sse = cell(l_length,1);
hist_Y_top = cell(l_length,1);

Y = predict(neural,images)';

for l = 1:l_length
    l_ind = l_inds(l);
    layer = neural.Layers(l_ind);
    layer.Weights = trans{1}(layer.Weights);
    [h,w,p,q] = size(layer.Weights);

    hist_delta{l} = zeros(maxsteps,h*w,1)*NaN;
    hist_coded{l} = zeros(maxsteps,h*w,1)*NaN;
    hist_W_sse{l} = zeros(maxsteps,h*w,1)*NaN;
    hist_Y_sse{l} = zeros(maxsteps,h*w,testsize)*NaN;
    hist_Y_top{l} = zeros(maxsteps,h*w,testsize)*NaN;
    
    for i = 1:h*w % iterate over the frequency bands
        [r,c] = ind2sub([h,w],i);
        scale = 2^floor(log2(sqrt(mean(layer.Weights(r,c,:).^2))/1024));
        coded = Inf;
        for j = 1:maxsteps
            % quantize each of the q slices
            quant = layer;
            delta = scale*sqrt(2^(j-1));
            quant.Weights(r,c,:) = quantize(quant.Weights(r,c,:),delta);
            coded = qentropy(quant.Weights(r,c,:))*(p*q);
            % assemble the net using layers
            quant.Weights = trans{2}(quant.Weights);
            ournet = replaceLayers(neural,quant);

            Y_hats = predict(ournet,images)';
            Y_cats = classify(nclass,Y_hats)';
            hist_Y_sse{l}(j,i,:) = mean((Y_hats - Y).^2);
            hist_Y_top{l}(j,i,:) = images.Labels == Y_cats;
            hist_W_sse{l}(j,i,1) = mean((quant.Weights(r,c,:) - neural.Layers(l_ind).Weights(r,c,:)).^2);
            hist_delta{l}(j,i,1) = delta;
            hist_coded{l}(j,i,1) = coded;

            disp(sprintf('%s %s | layer: %03d/%03d, band: %03d/%03d, scale: %3d, delta: %+5.1f, ymse: %5.2e, wmse: %5.2e, top1: %4.1f, rate: %5.2e', ...
                 archname, tranname, l, l_length, i, h*w, log2(scale), log2(delta), mean(hist_Y_sse{l}(j,i,:)), ...
                 hist_W_sse{l}(j,i,1), mean(hist_Y_top{l}(j,i,:))*100, coded/(p*q)));
            if coded == 0
                break
            end
        end
    end
end
%save([archname,'_',tranname,'_val_',num2str(testsize)],'hist_coded','hist_Y_sse','hist_Y_top','hist_delta','hist_W_sse');