function generate_RD_curves_intra(archname,tranname,testsize,inlayers,outlayer,strides)

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
    K = gettrans([tranname,'_intra'],archname,l);
    [h,w,p,q,g] = size(perm5(layers(l).Weights,layers(l)));
    layer_weights = reshape(transform_intra(layers(l).Weights,K{1}),[h*w,p,q,g]);
    hist_delta{l} = zeros(maxrates,maxsteps,h*w)*NaN;
    hist_coded{l} = zeros(maxrates,maxsteps,h*w)*NaN;
    hist_W_sse{l} = zeros(maxrates,maxsteps,h*w)*NaN;
    hist_Y_sse{l} = zeros(maxrates,maxsteps,h*w)*NaN;
    hist_Y_top{l} = zeros(maxrates,maxsteps,h*w)*NaN;
    s = strides(l);
    for i = 1:s:h*w % iterate over the frequency  bands
        rs = i:min(h*w,s+i-1);
        scale = floor(log2(sqrt(mean(reshape(layer_weights(rs,:,:,:),[],1).^2))));
        coded = Inf;
        offset = scale + 2;
        for k = 1:maxrates %number of bits
            B = k - 1;
            last_Y_sse = Inf;
            last_W_sse = Inf;
            for j = 1:maxsteps
                % quantize each of the q slices
                quant_weights = layer_weights;
                delta = offset + 0.25*(j-1);
                quant_weights(rs,:,:,:) = quantize(quant_weights(rs,:,:,:),2^delta,B);
                coded = B*(s*p*q*g); %qentropy(quant.Weights(r,c,:),B)*(p*q);
                % assemble the net using layers
                quant = layers(l);
                quant.Weights = transform_intra(reshape(quant_weights,[h,w,p,q,g]),K{2});
                ournet = replaceLayers(neural,quant);

                [Y_hats,Y_cats] = pred(ournet,nclass,images,outlayer);
                hist_Y_sse{l}(k,j,i) = mean((Y_hats(:) - Y(:)).^2);
                hist_Y_top{l}(k,j,i) = mean(images.Labels == Y_cats);
                hist_W_sse{l}(k,j,i) = mean((quant.Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2);
                hist_delta{l}(k,j,i) = delta;
                hist_coded{l}(k,j,i) = coded;
                mean_Y_sse = hist_Y_sse{l}(k,j,i);
                mean_W_sse = hist_W_sse{l}(k,j,i);
                disp(sprintf('%s %s | layer: %03d/%03d, band: %03d/%03d, scale: %3d, delta: %+6.2f, ymse: %5.2e, wmse: %5.2e, top1: %4.1f, rate: %5.2e', ...
                             archname, tranname, l, l_length, i, h*w, scale, delta, mean_Y_sse, ...
                             mean_W_sse, 100*hist_Y_top{l}(k,j,i), coded/(s*p*q*g)));
                if (mean_Y_sse > last_Y_sse) && ...
                   (mean_W_sse > last_W_sse) || ...
                   (B == 0)
                    [~,j] = min(hist_Y_sse{l}(k,:,i));
                    delta = hist_delta{l}(k,j,i);
                    offset = delta - 2;
                    break;
                end
                last_Y_sse = mean_Y_sse;
                last_W_sse = mean_W_sse;
            end
        end
    end
end
save(sprintf('%s_%s_val_%d_%s_intra',archname,tranname,testsize,outlayer),'hist_coded','hist_Y_sse','hist_Y_top','hist_delta','hist_W_sse','strides');