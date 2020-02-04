function generate_RD_curves_inter_base(archname,tranname,testsize,inlayers,outlayer,strides)

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
% archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
% tranname = 'idt2';
% testsize = 1024;
maxsteps = 16;
maxrates = 17;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
Y = pred(neural,images,outlayer);
Y_cats = getclass(neural,Y);

disp(sprintf('%s | top1: %5.2f', archname, 100*mean(images.Labels == Y_cats)));

load(sprintf('%s_cmeans_offset',archname));
layers = modifyConvLayers(neural,cmeans,offset);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers); 
l_length = length(l_kernel);

base_delta = cell(l_length,1);
base_coded = cell(l_length,1);
base_W_sse = cell(l_length,1);
base_Y_sse = cell(l_length,1);
base_Y_top = cell(l_length,1);

layers = neural.Layers(l_kernel);
for l = inlayers
    basis_vectors = gettrans([tranname,'_50000_inter'],archname,l);
    layer_weights = layers(l).Weights;%perm5(layers(l).Weights,layers(l),size(basis_vectors));
    [h,w,p,q,g] = size(layer_weights);
    layer_weights = reshape(basis_vectors(:,:,:,1)*reshape(permute(layer_weights,[3,5,4,1,2]),p*g,q*h*w),p*g,q*h*w);

    base_delta{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    base_coded{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    base_W_sse{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    base_Y_sse{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    base_Y_top{l} = zeros(maxrates,maxsteps,p*g)*NaN;
    s = strides(l);
    for i = 1:s:p*g % iterate over the frequency bands
        rs = i:min(p*g,s+i-1);
        scale = floor(log2(sqrt(mean(reshape(layer_weights(rs,:,:,:),[],1).^2))));
        if scale < -24 %all zeros
            continue
        end
        scale = floor(log2(sqrt(mean(reshape(basis_vectors(:,rs,:,2),[],1).^2))));
        coded = Inf;
        offset = scale - 2;
        for k = 1:maxrates %number of bits
            B = k - 1;
            for j = 1:maxsteps
                % quantize each of the q slices
                delta = offset + 0.25*(j-1);
                quant_vectors = basis_vectors;
                quant_vectors(:,rs,:,2) = quantize(quant_vectors(:,rs,:,2),2^delta,B);
                % assemble the net using layers
                quant = layers(l);
                quant.Weights = permute(reshape(quant_vectors(:,:,:,2)*layer_weights,p,g,q,h,w),[4,5,1,3,2]);
                coded = B*(s*1*p*g);
                base_delta{l}(k,j,i) = delta;
                base_coded{l}(k,j,i) = coded;
                base_W_sse{l}(k,j,i) = mean((quant.Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2);
            end

            [~,j] = min(base_W_sse{l}(k,:,i));
            delta = base_delta{l}(k,j,i);
            quant_vectors = basis_vectors;
            quant_vectors(:,rs,:,2) = quantize(quant_vectors(:,rs,:,2),2^delta,B);
            % assemble the net using layers
            quant = layers(l);
            quant.Weights = permute(reshape(quant_vectors(:,:,:,2)*layer_weights,p,g,q,h,w),[4,5,1,3,2]);
            offset = delta - 2;

            ournet = replaceLayers(neural,quant);
            tic; Y_hats = pred(ournet,images,outlayer); sec = toc;
            Y_cats = getclass(neural,Y_hats);
            base_Y_sse{l}(k,j,i) = mean((Y_hats(:) - Y(:)).^2);
            base_Y_top{l}(k,j,i) = mean(images.Labels == Y_cats);
            mean_Y_sse = base_Y_sse{l}(k,j,i);
            mean_Y_top = base_Y_top{l}(k,j,i);
            mean_W_sse = base_W_sse{l}(k,j,i);
            disp(sprintf('%s %s | layer: %03d/%03d, band: %04d/%04d, scale: %+6.2f,  delta: %+6.2f, ymse: %5.2e, wmse: %5.2e, top1: %4.1f, rate: %5.2e, time: %5.2fs', ...
                         archname, tranname, l, l_length, i, p*g, scale, delta, mean_Y_sse, mean_W_sse,...
                         100*mean(base_Y_top{l}(k,j,i)), coded/(s*1*p*g), sec));
        end
    end
    save(sprintf('%s_%s_val_%d_%d_%d_%s_inter_base',archname,tranname,testsize,l,l,outlayer),...
         '-v7.3','base_coded','base_Y_sse','base_Y_top','base_delta','base_W_sse','strides');
end
