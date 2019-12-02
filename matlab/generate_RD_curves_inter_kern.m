function generate_RD_curves_inter_kern(archname,tranname,testsize,inlayers,outlayer,strides)

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

kern_delta = cell(l_length,1);
kern_coded = cell(l_length,1);
kern_W_sse = cell(l_length,1);
kern_Y_sse = cell(l_length,1);
kern_Y_top = cell(l_length,1);

[Y,Y_cats] = pred(neural,nclass,images,outlayer);
disp(sprintf('%s | top1: %4.1f', archname, 100*mean(images.Labels == Y_cats)));

layers = neural.Layers(l_kernel);
for l = inlayers
    [h,w,p,q,g] = size(layers(l).Weights);
    basis_vectors = gettrans([tranname,'_inter'],archname,l);
    layer_weights = reshape(permute(transform_inter(layers(l).Weights,basis_vectors(:,:,:,1)),[1,2,3,5,4]),[h,w,p*g,q]);
    kern_delta{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    kern_coded{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    kern_W_sse{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    kern_Y_sse{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    kern_Y_top{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    s = strides(l);
    for i = 1:s:p*g % iterate over the frequency bands
        rs = i:min(p*g,s+i-1);
        scale = floor(log2(sqrt(mean(reshape(layer_weights(:,:,rs,:),[],1).^2))));
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
                quant_weights(:,:,rs,:) = quantize(quant_weights(:,:,rs,:),2^delta,B);
                coded = B*(s*h*w*q);
                % assemble the net using layers
                quant = layers(l);
                quant.Weights = transform_inter(permute(reshape(quant_weights,[h,w,p,g,q]),[1,2,3,5,4]),basis_vectors(:,:,:,2));
                ournet = replaceLayers(neural,quant);

                [Y_hats,Y_cats] = pred(ournet,nclass,images,outlayer);
                kern_Y_sse{l}(k,j,i) = mean((Y_hats(:) - Y(:)).^2);
                kern_Y_top{l}(k,j,i) = mean(images.Labels == Y_cats);
                kern_W_sse{l}(k,j,i) = mean((quant.Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2);
                kern_delta{l}(k,j,i) = delta;
                kern_coded{l}(k,j,i) = coded;
                mean_Y_sse = kern_Y_sse{l}(k,j,i);
                mean_W_sse = kern_W_sse{l}(k,j,i);
                disp(sprintf('%s %s | layer: %03d/%03d, band: %03d/%03d, scale: %3d, delta: %+6.2f, ymse: %5.2e, wmse: %5.2e, top1: %4.1f, rate: %5.2e', ...
                             archname, tranname, l, l_length, i, p*g, scale, delta, mean_Y_sse, ...
                             mean_W_sse, 100*mean(kern_Y_top{l}(k,j,i)), coded/(s*h*w*q)));
                if (mean_Y_sse > last_Y_sse) && ...
                   (mean_W_sse > last_W_sse) || ...
                   (B == 0)
                    [~,j] = min(kern_Y_sse{l}(k,:,i));
                    delta = kern_delta{l}(k,j,i);
                    offset = delta - 2;
                    break;
                end
                last_Y_sse = mean_Y_sse;
                last_W_sse = mean_W_sse;
            end
        end
    end
end
save(sprintf('%s_%s_val_%d_%s_inter',archname,tranname,testsize,outlayer),'kern_coded','kern_Y_sse','kern_Y_top','kern_delta','kern_W_sse','strides');