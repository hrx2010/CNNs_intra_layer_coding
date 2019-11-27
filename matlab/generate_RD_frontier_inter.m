function generate_RD_frontier_inter(archname,tranname,testsize,inlayers,outlayer,strides)

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
% archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';

maxsteps = 96;

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
    ydist = cell(l_length,1);
    coded = cell(l_length,1);
    delta = cell(l_length,1);
    wdist = cell(l_length,1);
    denom = cell(l_length,1);

    quants = neural.Layers(l_kernel);
    for l = inlayers
        K = gettrans([tranname,'_inter'],archname,l);
        [h,w,p,q,g] = size(quants(l).Weights);
        quant_weights = reshape(permute(transform_inter(quants(l).Weights,K{1}),[1,2,3,5,4]),[h,w,p*g,q]);
        [best_Y_sse,best_delta,best_coded] = finddelta(mean(hist_Y_sse{l},4),hist_delta{l},hist_coded{l});
        ydist{l} = lambda2points(best_coded,best_Y_sse,best_Y_sse,2^slope);
        coded{l} = lambda2points(best_coded,best_Y_sse,best_coded,2^slope);
        delta{l} = lambda2points(best_coded,best_Y_sse,best_delta,2^slope);
        denom{l} = h*w*p*q*g;%ones(size(coded{l}))*(h*w*q);
        s = strides(l);
        for i = 1:s:p*g
            rs = i:min(p*g,s+i-1);
            % quantize for the given lambda
            quant_weights(:,:,rs,:) = quantize(quant_weights(:,:,rs,:),2^delta{l}(i),coded{l}(i)/(s*h*w*q));
        end
        quants(l).Weights = transform_inter(permute(reshape(quant_weights,[h,w,p,g,q]),[1,2,3,5,4]),K{2});
        wdist{l} = double(sum((quants(l).Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2));
    end
    ournet = replaceLayers(neural,quants);

    ydist = cell2mat(ydist);
    coded = cell2mat(coded);
    delta = cell2mat(delta);
    wdist = cell2mat(wdist);
    denom = cell2mat(denom);
    
    [Y_hats,Y_cats] = pred(ournet,nclass,images,outlayer);
    hist_sum_Y_sse(j,1) = mean((Y_hats(:) - Y(:)).^2,1);
    hist_sum_Y_top(j,1) = mean(images.Labels == Y_cats);
    pred_sum_Y_sse(j,1) = sum(ydist(:),'omitnan');
    hist_sum_W_sse(j,1) = sum(wdist(:),'omitnan')/sum(denom(:),'omitnan');
    hist_sum_coded(j,1) = sum(coded(:),'omitnan')/sum(denom(:),'omitnan');

    disp(sprintf('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %4.1f, rate: %5.2e',...
                 archname, tranname, slope, hist_sum_Y_sse(j,1), pred_sum_Y_sse(j,1), ...
                 hist_sum_W_sse(j,1), 100*hist_sum_Y_top(j,1), hist_sum_coded(j,1)));
    if hist_sum_coded(j) == 0
        break;
    end
end

save(sprintf('%s_%s_sum_%d_%d_%d_%s_inter',archname,tranname,testsize,inlayers(1),inlayers(end),outlayer),...
     'hist_sum_coded','hist_sum_Y_sse','pred_sum_Y_sse','hist_sum_W_sse','hist_sum_Y_top');