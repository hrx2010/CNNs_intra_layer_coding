function generate_RD_curves_joint_kern(archname,tranname,testsize,inlayers,outlayer,strides)

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
Y = exp(Y)./sum(exp(Y));

disp(sprintf('%s | top1: %5.2f', archname, 100*mean(images.Labels == Y_cats)));

load(sprintf('%s_cmeans_offset',archname));
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
    basis_vectors = gettrans([tranname,'_50000_joint'],archname,l);
    [h,w,p,q,g] = size(perm5(layers(l).Weights,layers(l),size(basis_vectors,1)));
    layer_weights = reshape(permute(transform_inter(perm5(layers(l).Weights,layers(l),size(basis_vectors,1)),...
                                                    basis_vectors(:,:,:,1)),[1,2,3,5,4]),[1,1,h*w*p*g,q]);
    kern_delta{l} = zeros(maxrates,maxsteps,h*w*p*g)*NaN;
    kern_coded{l} = zeros(maxrates,maxsteps,h*w*p*g)*NaN;
    kern_W_sse{l} = zeros(maxrates,maxsteps,h*w*p*g)*NaN;
    kern_Y_sse{l} = zeros(maxrates,maxsteps,h*w*p*g)*NaN;
    kern_Y_top{l} = zeros(maxrates,maxsteps,h*w*p*g)*NaN;
    s = strides(l);
    for i = 1:s:h*w*p*g % iterate over the frequency bands
        rs = i:min(h*w*p*g,s+i-1);
        scale = floor(log2(sqrt(mean(reshape(layer_weights(:,:,rs,:),[],1).^2))));
        if scale < -24 %all zeros
            continue
        end
        scale = floor(log2(sqrt(mean(reshape(layer_weights(:,:,rs,:),[],1).^2))));
        coded = Inf;
        offset = scale + 2;
        for k = 1:maxrates %number of bits
            B = k - 1;
            for j = 1:maxsteps
                % quantize each of the q slices
                delta = offset + 0.25*(j-1);
                quant_weights = layer_weights;
                quant_weights(:,:,rs,:) = quantize(quant_weights(:,:,rs,:),2^delta,B);
                % assemble the net using layers
                quant = layers(l);
                delta_weights = reshape(quant_weights(:,:,rs,:) - layer_weights(:,:,rs,:),s,[]); 
                quant.Weights = perm5(permute(reshape(reshape(basis_vectors(:,rs,:,2),[],s)*delta_weights,...
                                     [h,w,p,g,q]),[1,2,3,5,4]),quant,size(basis_vectors,1))+quant.Weights;         
                % quant.Weights = perm5(transform_inter(permute(reshape(quant_weights,[h,w,p,g,q]),[1,2,3,5,4]),...
                %                                      basis_vectors(:,:,:,2)),quant,size(basis_vectors,1));
                coded = B*(s*1*1*q);
                kern_delta{l}(k,j,i) = delta;
                kern_coded{l}(k,j,i) = coded;
                kern_W_sse{l}(k,j,i) = mean((quant.Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2);
            end

            [~,j] = min(kern_W_sse{l}(k,:,i));
            delta = kern_delta{l}(k,j,i);
            quant_weights = layer_weights;
            quant_weights(:,:,rs,:) = quantize(quant_weights(:,:,rs,:),2^delta,B);
            % assemble the net using layers
            quant = layers(l);
            delta_weights = reshape(quant_weights(:,:,rs,:) - layer_weights(:,:,rs,:),s,q); 
            quant.Weights = perm5(permute(reshape(reshape(basis_vectors(:,rs,:,2),[],s)*delta_weights,...
                                 [h,w,p,g,q]),[1,2,3,5,4]),quant,size(basis_vectors,1))+quant.Weights;         
            % quant.Weights = perm5(transform_inter(permute(reshape(quant_weights,[h,w,p,g,q]),[1,2,3,5,4]),...
            %                                      basis_vectors(:,:,:,2)),quant,size(basis_vectors,1));
            offset = delta - 2;

            ournet = replaceLayers(neural,quant);
            tic; Y_hats = pred(ournet,images,outlayer); sec = toc;
            Y_cats = getclass(neural,Y_hats);
            Y_hats = exp(Y_hats)./sum(exp(Y_hats));  
            kern_Y_sse{l}(k,j,i) = mean((Y_hats(:) - Y(:)).^2);
            kern_Y_top{l}(k,j,i) = mean(images.Labels == Y_cats);
            mean_Y_sse = kern_Y_sse{l}(k,j,i);
            mean_Y_top = kern_Y_top{l}(k,j,i);
            mean_W_sse = kern_W_sse{l}(k,j,i);
            disp(sprintf('%s %s | layer: %03d/%03d, band: %04d/%04d, scale: %+6.2f, delta: %+6.2f, ymse: %5.2e, wmse: %5.2e, top1: %5.2f, rate: %2.0f, time: %5.2fs', ...
                         archname, tranname, l, l_length, i, h*w*p*g, scale, delta, mean_Y_sse, mean_W_sse,...
                         100*mean(kern_Y_top{l}(k,j,i)), coded/(s*1*1*q), sec));
        end
    end
    save(sprintf('%s_%s_val_%d_%d_%d_%s_joint_kern',archname,tranname,testsize,l,l,outlayer),...
         'kern_coded','kern_Y_sse','kern_Y_top','kern_delta','kern_W_sse','strides');
end
