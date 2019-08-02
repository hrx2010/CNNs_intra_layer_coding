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
maxrates = 8;

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

Y = pred(neural,nclass,images);

layers = neural.Layers(l_kernel);
for l = 1:l_length
    layer = layers(l);
    layer.Weights = trans{1}(layer.Weights);
    [h,w,p,q] = size(layer.Weights);

    hist_delta{l} = zeros(maxrates,maxsteps,h*w,1)*NaN;
    hist_coded{l} = zeros(maxrates,maxsteps,h*w,1)*NaN;
    hist_W_sse{l} = zeros(maxrates,maxsteps,h*w,1)*NaN;
    hist_Y_sse{l} = zeros(maxrates,maxsteps,h*w,testsize)*NaN;
    hist_Y_top{l} = zeros(maxrates,maxsteps,h*w,testsize)*NaN;
    
    for i = 1:h*w % iterate over the frequency bands
        [r,c] = ind2sub([h,w],i);
        scale = 2^floor(log2(sqrt(mean(layer.Weights(r,c,:).^2))));
        coded = Inf;
        for k = 1:maxrates %number of bits
            B = k;
            for j = 1:maxsteps
                % quantize each of the q slices
                quant = layer;
                delta = scale*(2^(j-10));
                quant.Weights(r,c,:) = quantize(quant.Weights(r,c,:),delta,B);
                coded = qentropy(quant.Weights(r,c,:),B)*(p*q);
                % assemble the net using layers
                quant.Weights = trans{2}(quant.Weights);
                ournet = replaceLayers(neural,quant);

                [Y_hats,Y_cats] = pred(ournet,nclass,images);
                hist_Y_sse{l}(k,j,i,:) = mean((Y_hats - Y).^2);
                hist_Y_top{l}(k,j,i,:) = images.Labels == Y_cats;
                hist_W_sse{l}(k,j,i,1) = mean((quant.Weights(r,c,:) - neural.Layers(l_kernel(l)).Weights(r,c,:)).^2);
                hist_delta{l}(k,j,i,1) = delta;
                hist_coded{l}(k,j,i,1) = coded;

                disp(sprintf('%s %s | layer: %03d/%03d, band: %03d/%03d, scale: %3d, delta: %+5.1f, ymse: %5.2e, wmse: %5.2e, top1: %4.1f, rate: %5.2e', ...
                             archname, tranname, l, l_length, i, h*w, log2(scale), log2(delta), mean(hist_Y_sse{l}(k,j,i,:)), ...
                             hist_W_sse{l}(k,j,i,1), 100*mean(hist_Y_top{l}(k,j,i,:)), coded/(p*q)));
                if coded == 0
                    break
                end
            end
        end
    end
end
save(sprintf('%s_%s_val_%d',archname,tranname,testsize),'hist_coded','hist_Y_sse','hist_Y_top','hist_delta','hist_W_sse');