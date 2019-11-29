function generate_RD_frontier_inter_total(archname,tranname,testsize,inlayers,outlayer,strides)

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
% archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';

maxsteps = 96;
load(sprintf('%s_%s_val_100_%s_inter_basis',archname,tranname,outlayer));
load(sprintf('%s_%s_val_100_%s_inter',archname,tranname,outlayer));
[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);

[cmeans,offset] = channelMeans(neural,images);
layers = modifyConvLayers(layers,cmeans,offset);
neural = assembleNetwork(layers);
nclass = assembleNetwork(lclass);

l_kernel = findconv(neural.Layers); % or specify the layer number directly
l_length = length(l_kernel);

hist_sum_Y_top = zeros(maxsteps,1)*NaN;
hist_sum_Y_sse = zeros(maxsteps,1)*NaN;
pred_sum_Y_sse = zeros(maxsteps,1)*NaN;
hist_sum_coded = zeros(maxsteps,1)*NaN;
hist_sum_W_sse = zeros(maxsteps,1)*NaN;

[Y,Y_cats] = pred(neural,nclass,images,outlayer);
disp(sprintf('%s | top1: %4.1f', archname, 100*mean(images.Labels == Y_cats)));

for j = 1:maxsteps
    slope = -48 + 0.50*(j-1);
    kern_ydist = cell(l_length,1);
    kern_coded = cell(l_length,1);
    kern_delta = cell(l_length,1);
    kern_wdist = cell(l_length,1);
    kern_denom = cell(l_length,1);

    base_ydist = cell(l_length,1);
    base_coded = cell(l_length,1);
    base_delta = cell(l_length,1);
    base_wdist = cell(l_length,1);
    base_denom = cell(l_length,1);

    quants = neural.Layers(l_kernel);
    for l = inlayers
        [h,w,p,q,g] = size(quants(l).Weights);
        quant_vectors = gettrans([tranname,'_inter'],archname,l);
        quant_weights = reshape(permute(transform_inter(quants(l).Weights,K{1}),[1,2,3,5,4]),[h,w,p*g,q]);
        [kern_best_Y_sse,kern_best_delta,kern_best_coded] = finddelta(mean(kern_Y_sse{l},4),kern_delta{l},kern_coded{l});
        kern_ydist{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_Y_sse,2^slope);
        kern_coded{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_coded,2^slope);
        kern_delta{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_delta,2^slope);
        kern_denom{l} = h*w*p*q*g;%ones(size(coded{l}))*(h*w*q);

        [base_best_Y_sse,base_best_delta,base_best_coded] = finddelta(mean(base_Y_sse{l},4),base_delta{l},base_coded{l});
        base_ydist{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_Y_sse,2^slope);
        base_coded{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_coded,2^slope);
        base_delta{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_delta,2^slope);
        base_denom{l} = 1*1*p*p*g;

        s = strides(l);
        for i = 1:s:p*g
            rs = i:min(p*g,s+i-1);
            % quantize for the given lambda
            quant_weights(:,:,rs,:) = quantize(quant_weights(:,:,rs,:),2^delta_kern{l}(i),coded_kern{l}(i)/(s*h*w*q));
            quant_vectors(:,rs,:,2) = quantize(quant_vectors(:,rs,:,2),2^delta_base{l}(i),coded_base{l}(i)/(s*1*1*p));
        end
        quants(l).Weights = transform_inter(permute(reshape(quant_weights,[h,w,p,g,q]),[1,2,3,5,4]),quant_vectors(:,:,:,2));
        kern_wdist{l} = double(sum((quants(l).Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2));
    end
    ournet = replaceLayers(neural,quants);

    kern_ydist = cell2mat(kern_ydist);
    kern_coded = cell2mat(kern_coded);
    kern_delta = cell2mat(kern_delta);
    kern_wdist = cell2mat(kern_wdist);
    kern_denom = cell2mat(kern_denom);

    base_ydist = cell2mat(base_ydist);
    base_coded = cell2mat(base_coded);
    base_delta = cell2mat(base_delta);
    base_wdist = cell2mat(base_wdist);
    base_denom = cell2mat(base_denom);
    
    [Y_hats,Y_cats] = pred(ournet,nclass,images,outlayer);
    hist_sum_Y_sse(j,1) = mean((Y_hats(:) - Y(:)).^2,1);
    hist_sum_Y_top(j,1) = mean(images.Labels == Y_cats);
    pred_sum_Y_sse(j,1) = (sum(kern_ydist(:),'omitnan') + sum(base_ydist(:),'omitnan'));
    hist_sum_W_sse(j,1) = (sum(kern_wdist(:),'omitnan') + sum(base_wdist(:),'omitnan'))...
                        / (sum(kern_denom(:),'omitnan') + sum(base_denom(:),'omitnan'));
    hist_sum_coded(j,1) = (sum(kern_coded(:),'omitnan') + sum(base_coded(:),'omitnan'))...
                        / (sum(kern_denom(:),'omitnan') + sum(base_denom(:),'omitnan'));

    disp(sprintf('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %4.1f, rate: %5.2e',...
                 archname, tranname, slope, hist_sum_Y_sse(j,1), pred_sum_Y_sse(j,1), ...
                 hist_sum_W_sse(j,1), 100*hist_sum_Y_top(j,1), hist_sum_coded(j,1)));
    if hist_sum_coded(j) == 0
        break;
    end
end

save(sprintf('%s_%s_sum_%d_%d_%d_%s_inter_total',archname,tranname,testsize,inlayers(1),inlayers(end),outlayer),...
     'hist_sum_coded','hist_sum_Y_sse','pred_sum_Y_sse','hist_sum_W_sse','hist_sum_Y_top');