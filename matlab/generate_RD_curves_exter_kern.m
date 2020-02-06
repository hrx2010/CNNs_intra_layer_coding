function generate_RD_curves_exter_kern(archname,tranname,testsize,inlayers,outlayer,strides)

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

disp(sprintf('%s | top1: %4.1f', archname, 100*mean(images.Labels == Y_cats)));

load(sprintf('%s_cmeans_offset',archname));
load(sprintf('%s_dims',archname));
layers = modifyConvLayers(neural,cmeans,offset);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers); 
l_length = length(l_kernel);

kern_delta = cell(l_length,1);
kern_coded = cell(l_length,1);
kern_W_sse = cell(l_length,1);
kern_Y_sse = cell(l_length,1);
kern_Y_top = cell(l_length,1);

layers = neural.Layers(l_kernel);
for l = inlayers
    basis_vectors = gettrans([tranname,'_50000_exter'],archname,l);
    layer_weights = layers(l).Weights;
    [h,w,p,q,g] = size(layer_weights);
    layer_weights = reshape(basis_vectors(:,:,:,1)*reshape(permute(layer_weights,[4,3,5,1,2]),q,p*g*h*w),q,p*g*h*w);

    kern_delta{l} = zeros(maxrates,maxsteps,1*1*1*q)*NaN;
    kern_coded{l} = zeros(maxrates,maxsteps,1*1*1*q)*NaN;
    kern_W_sse{l} = zeros(maxrates,maxsteps,1*1*1*q)*NaN;
    kern_Y_sse{l} = zeros(maxrates,maxsteps,1*1*1*q)*NaN;
    kern_Y_top{l} = zeros(maxrates,maxsteps,1*1*1*q)*NaN;
    s = strides(l);
    for i = 1:s:1*1*1*q % iterate over the frequency bands
        rs = i:min(1*1*1*q,s+i-1);
        scale = floor(log2(sqrt(mean(reshape(layer_weights(rs,:,:,:),[],1).^2))));
        if scale < -24 %all zeros
            continue
        end
        scale = floor(log2(sqrt(mean(reshape(layer_weights(rs,:,:,:),[],1).^2))));
        coded = Inf;
        offset = scale + 2;
        for k = 1:maxrates %number of bits
            B = k - 1;
            for j = 1:maxsteps
                % quantize each of the q slices
                delta = offset + 0.25*(j-1);
                quant_weights = layer_weights;
                quant_weights(rs,:,:,:) = quantize(quant_weights(rs,:,:,:),2^delta,B);
                % assemble the net using layers
                quant = layers(l);
                quant.Weights = permute(reshape(basis_vectors(:,:,:,2)*quant_weights,[q,p,g,h,w]),[4,5,2,1,3]);
                coded = B*(length(rs)*p*g*h*w);
                kern_delta{l}(k,j,i) = delta;
                kern_coded{l}(k,j,i) = coded;
                kern_W_sse{l}(k,j,i) = mean((quant.Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2);
            end

            [~,j] = min(kern_W_sse{l}(k,:,i));
            delta = kern_delta{l}(k,j,i);
            quant_weights = layer_weights;
            quant_weights(rs,:,:,:) = quantize(quant_weights(rs,:,:,:),2^delta,B);
            % assemble the net using layers
            quant = layers(l);
            quant.Weights = permute(reshape(basis_vectors(:,:,:,2)*quant_weights,[q,p,g,h,w]),[4,5,2,1,3]);
            offset = delta - 2;

            ournet = replaceLayers(neural,quant);
            tic; Y_hats = pred(ournet,images,outlayer); sec = toc;
            Y_cats = getclass(neural,Y_hats);
            kern_Y_sse{l}(k,j,i) = mean((Y_hats(:) - Y(:)).^2);
            kern_Y_top{l}(k,j,i) = mean(images.Labels == Y_cats);
            mean_Y_sse = kern_Y_sse{l}(k,j,i);
            mean_Y_top = kern_Y_top{l}(k,j,i);
            mean_W_sse = kern_W_sse{l}(k,j,i);
            disp(sprintf('%s %s | layer: %03d/%03d, band: %04d/%04d, delta: %+6.2f, ymse: %5.2e, wmse: %5.2e, top1: %4.1f, rate: %5.2e, time: %5.2fs', ...
                         archname, tranname, l, l_length, i, 1*q, delta, mean_Y_sse, mean_W_sse,...
                         100*mean(kern_Y_top{l}(k,j,i)), coded/(length(rs)*p*g*h*w), sec));
        end
    end
    save(sprintf('%s_%s_val_%d_%d_%d_%s_inter_kern',archname,tranname,testsize,l,l,outlayer),...
         '-v7.3','kern_coded','kern_Y_sse','kern_Y_top','kern_delta','kern_W_sse','strides');
end
