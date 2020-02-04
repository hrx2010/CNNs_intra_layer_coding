function generate_RD_curves_randc_kern(archname,tranname,testsize,inlayers,outlayer,strides)

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
% Y = exp(Y)./sum(exp(Y));

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
    exter_vectors = gettrans([tranname,'_50000_exter'],archname,l);
    inter_vectors = gettrans([tranname,'_50000_inter'],archname,l);
    layer_weights = layers(l).Weights;%perm5(layers(l).Weights,layers(l),size(basis_vectors));
    [h,w,p,q,g] = size(layer_weights);
    layer_weights = reshape(inter_vectors(:,:,:,1)*reshape(permute(layer_weights,[3,5,4,1,2]),p*g,q*h*w),p,g,q,h,w);
    layer_weights = reshape(exter_vectors(:,:,:,1)*reshape(permute(layer_weights,[3,1,2,4,5]),q,p*g*h*w),q,p*g,h*w);

    [s,t] = deal(2*strides(l,1),2*strides(l,2));
    points = ceil([q*1/s,p*g/t]);
    kern_delta{l} = zeros(maxrates,maxsteps,points(1),points(2))*NaN;
    kern_coded{l} = zeros(maxrates,maxsteps,points(1),points(2))*NaN;
    kern_W_sse{l} = zeros(maxrates,maxsteps,points(1),points(2))*NaN;
    kern_Y_sse{l} = zeros(maxrates,maxsteps,points(1),points(2))*NaN;
    kern_Y_top{l} = zeros(maxrates,maxsteps,points(1),points(2))*NaN;
    for r = 1:s:q*1
        for c = 1:t:p*g
            rs = r:min(q*1,s+r-1);
            cs = c:min(p*g,t+c-1);
            ri = (r-1)/s+1;
            ci = (c-1)/t+1;
            scale = floor(log2(sqrt(mean(reshape(layer_weights(rs,cs,:),[],1).^2))));
            if scale < -24 %all zeros
                continue
            end
            scale = floor(log2(sqrt(mean(reshape(layer_weights(rs,cs,:),[],1).^2))));
            coded = Inf;
            offset = scale + 2;
            for k = 1:maxrates %number of bits
                B = k - 1;
                for j = 1:maxsteps
                    % quantize each of the q slices
                    delta = offset + 0.25*(j-1);
                    quant_weights = layer_weights;
                    quant_weights(rs,cs,:) = quantize(quant_weights(rs,cs,:),2^delta,B);
                    % assemble the net using layers
                    quant_weights = permute(reshape(exter_vectors(:,:,:,2)*reshape(quant_weights,q,p*g*h*w),q,p,g,h,w),[2,3,1,4,5]);
                    quant_weights = permute(reshape(inter_vectors(:,:,:,2)*reshape(quant_weights,p*g,q*h*w),p,g,q,h,w),[4,5,1,3,2]);
                    coded = B*(length(rs)*length(cs)*1*h*w);
                    kern_delta{l}(k,j,ri,ci) = delta;
                    kern_coded{l}(k,j,ri,ci) = coded;
                    kern_W_sse{l}(k,j,ri,ci) = mean((quant_weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2);
                end
                
                [~,j] = min(kern_W_sse{l}(k,:,ri,ci));
                delta = kern_delta{l}(k,j,ri,ci);
                quant_weights = layer_weights;
                quant_weights(rs,cs,:) = quantize(quant_weights(rs,cs,:),2^delta,B);
                quant_weights = permute(reshape(exter_vectors(:,:,:,2)*reshape(quant_weights,q,p*g*h*w),q,p,g,h,w),[2,3,1,4,5]);
                quant_weights = permute(reshape(inter_vectors(:,:,:,2)*reshape(quant_weights,p*g,q*h*w),p,g,q,h,w),[4,5,1,3,2]);
                quant = layers(l);
                quant.Weights = quant_weights;
                offset = delta - 2;
                
                ournet = replaceLayers(neural,quant);
                tic; Y_hats = pred(ournet,images,outlayer); sec = toc;
                Y_cats = getclass(neural,Y_hats);
                % Y_hats = exp(Y_hats)./sum(exp(Y_hats));  
                kern_Y_sse{l}(k,j,ri,ci) = mean((Y_hats(:) - Y(:)).^2);
                kern_Y_top{l}(k,j,ri,ci) = mean(images.Labels == Y_cats);
                mean_Y_sse = kern_Y_sse{l}(k,j,ri,ci);
                mean_Y_top = kern_Y_top{l}(k,j,ri,ci);
                mean_W_sse = kern_W_sse{l}(k,j,ri,ci);
                i = sub2ind(points([2,1]),ci,ri);
                disp(sprintf('%s %s | layer: %03d/%03d, band: %04d/%04d, scale: %+6.2f, delta: %+6.2f, ymse: %5.2e, wmse: %5.2e, top1: %5.2f, rate: %2.0f, time: %5.2fs', ...
                             archname, tranname, l, l_length, i, prod(points), scale, delta, mean_Y_sse, mean_W_sse,...
                             100*mean(kern_Y_top{l}(k,j,ri,ci)), coded/(length(rs)*length(cs)*1*h*w), sec));
            end
            save(sprintf('%s_%s_val_%d_%d_%d_%s_randc_kern',archname,tranname,testsize,l,l,outlayer),...
                 '-v7.3','kern_coded','kern_Y_sse','kern_Y_top','kern_delta','kern_W_sse','strides');
        end
    end
end