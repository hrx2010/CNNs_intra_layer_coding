function generate_RD_curves_intra(archname,tranname,testsize,inlayers,outlayer)

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
% archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
% tranname = 'idt2';
% testsize = 1024;
maxsteps = 32;
maxrates = 17;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);

[cmeans,offset] = channelMeans(neural,images);
layers = modifyConvLayers(layers,cmeans,offset);
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

[Y,Y_cats] = pred(neural,nclass,images,outlayer);
disp(sprintf('%s | top1: %4.1f', archname, 100*mean(images.Labels == Y_cats)));

layers = neural.Layers(l_kernel);
for l = inlayers
    layer = layers(l);
    K = gettrans(tranname,archname,l);
    [h,w,p,q] = size(layer.Weights);
    layer.Weights = transform(layer.Weights,K{1});
    hist_delta{l} = zeros(maxrates,maxsteps,h*w,1)*NaN;
    hist_coded{l} = zeros(maxrates,maxsteps,h*w,1)*NaN;
    hist_W_sse{l} = zeros(maxrates,maxsteps,h*w,1)*NaN;
    hist_Y_sse{l} = zeros(maxrates,maxsteps,h*w,testsize)*NaN;
    hist_Y_top{l} = zeros(maxrates,maxsteps,h*w,testsize)*NaN;
    
    for i = 1:h*w % iterate over the frequency bands
        [r,c] = ind2sub([h,w],i);
        scale = floor(log2(sqrt(mean(reshape(layer.Weights(r,c,:),[],1).^2))));
        coded = Inf;
        offset = scale + 2;
        for k = 1:maxrates %number of bits
            B = k - 1;
            last_Y_sse = Inf;
            last_W_sse = Inf;
            for j = 1:maxsteps
                % quantize each of the q slices
                quant = layer;
                delta = offset + 0.25*(j-1);
                quant.Weights(r,c,:) = quantize(quant.Weights(r,c,:),2^delta,B);
                coded = B*(p*q); %qentropy(quant.Weights(r,c,:),B)*(p*q);
                % assemble the net using layers
                quant.Weights = transform(quant.Weights,K{2});
                ournet = replaceLayers(neural,quant);

                [Y_hats,Y_cats] = pred(ournet,nclass,images,outlayer);
                hist_Y_sse{l}(k,j,i,:) = mean((Y_hats - Y).^2);
                hist_Y_top{l}(k,j,i,:) = images.Labels == Y_cats;
                hist_W_sse{l}(k,j,i,1) = mean((quant.Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2);
                hist_delta{l}(k,j,i,1) = delta;
                hist_coded{l}(k,j,i,1) = coded;
                mean_Y_sse = mean(hist_Y_sse{l}(k,j,i,:));
                mean_W_sse = mean(hist_W_sse{l}(k,j,i,1));
                disp(sprintf('%s %s | layer: %03d/%03d, band: %03d/%03d, scale: %3d, delta: %+6.2f, ymse: %5.2e, wmse: %5.2e, top1: %4.1f, rate: %5.2e', ...
                             archname, tranname, l, l_length, i, h*w, scale, delta, mean_Y_sse, ...
                             mean_W_sse, 100*mean(hist_Y_top{l}(k,j,i,:)), coded/(p*q)));
                if (mean_Y_sse > last_Y_sse) && ...
                   (mean_W_sse > last_W_sse) || ...
                   (B == 0)
                    [~,j] = min(mean(hist_Y_sse{l}(k,:,i,:),4));
                    delta = hist_delta{l}(k,j,i,1);
                    offset = delta - 2;
                    break;
                end
                last_Y_sse = mean_Y_sse;
                last_W_sse = mean_W_sse;
            end
        end
    end
end
save(sprintf('%s_%s_val_%d_%s_intra',archname,tranname,testsize,outlayer),'hist_coded','hist_Y_sse','hist_Y_top','hist_delta','hist_W_sse');