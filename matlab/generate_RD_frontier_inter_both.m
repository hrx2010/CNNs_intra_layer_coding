function generate_RD_frontier_inter_both(archname,tranname,testsize,inlayers,outlayer,strides)

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
% archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';

maxsteps = 96;
load(sprintf('%s_cmeans_offset',archname));

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
Y = pred(neural,images,outlayer);
Y_cats = getclass(neural,Y);

disp(sprintf('%s | top1: %4.1f', archname, 100*mean(images.Labels == Y_cats)));

layers = modifyConvLayers(neural,cmeans,offset);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers,{'conv'}); % or specify the layer number directly
l_length = length(l_kernel);

hist_sum_Y_top = zeros(maxsteps,1)*NaN;
hist_sum_Y_sse = zeros(maxsteps,1)*NaN;
pred_sum_Y_sse = zeros(maxsteps,1)*NaN;
hist_sum_coded = zeros(maxsteps,1)*NaN;
hist_sum_W_sse = zeros(maxsteps,1)*NaN;
hist_sum_denom = zeros(maxsteps,l_length)*NaN;
hist_sum_non0s = zeros(maxsteps,l_length)*NaN;
hist_sum_total = zeros(maxsteps,l_length)*NaN;

for j = 1:maxsteps
    slope = -32 + 0.50*(j-1);
    ydist_kern = cell(l_length,1);
    coded_kern = cell(l_length,1);
    delta_kern = cell(l_length,1);
    wdist_kern = cell(l_length,1);
    denom_kern = cell(l_length,1);
    non0s_kern = cell(l_length,1);

    ydist_base = cell(l_length,1);
    coded_base = cell(l_length,1);
    delta_base = cell(l_length,1);
    wdist_base = cell(l_length,1);
    denom_base = cell(l_length,1);
    non0s_base = cell(l_length,1);

    quants = neural.Layers(l_kernel);
    for l = inlayers
        load(sprintf('%s_%s_val_500_%d_%d_%s_inter_base',archname,tranname,l,l,outlayer));
        load(sprintf('%s_%s_val_500_%d_%d_%s_inter_kern',archname,tranname,l,l,outlayer));
        quant_vectors = gettrans([tranname,'_50000_inter'],archname,l);
        [h,w,p,q,g] = size(quants(l).Weights);
        quant_weights = quants(l).Weights;
        quant_weights = reshape(quant_vectors(:,:,:,1)*reshape(permute(quant_weights,[3,5,4,1,2]),p*g,q*h*w),p*g,q*h*w);
        [kern_best_Y_sse,kern_best_delta,kern_best_coded] = finddelta(mean(kern_Y_sse{l},4),kern_delta{l},kern_coded{l});
        ydist_kern{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_Y_sse,2^slope);
        coded_kern{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_coded,2^slope);
        delta_kern{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_delta,2^slope);
        denom_kern{l} = h*w*p*q*g;%ones(size(coded{l}))*(h*w*q);

        [base_best_Y_sse,base_best_delta,base_best_coded] = finddelta(mean(base_Y_sse{l},4),base_delta{l},base_coded{l});
        ydist_base{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_Y_sse,2^slope);
        coded_base{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_coded,2^slope);
        delta_base{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_delta,2^slope);
        denom_base{l} = 1*p*g*p*g;

        s = strides(l);
        for i = 1:s:p*g
            rs = i:min(p*g,s+i-1);
            scale = floor(log2(sqrt(mean(reshape(quant_weights(rs,:,:,:),[],1).^2))));
            if scale < -24 %all zeros
                quant_weights(rs,:,:,:) = 0; %actual zero
                continue
            end
            % quantize for the given lambda
            quant_weights(rs,:,:,:) = quantize(quant_weights(rs,:,:,:),2^delta_kern{l}(i),coded_kern{l}(i)/(length(rs)*h*w*q));
            quant_vectors(:,rs,:,2) = quantize(quant_vectors(:,rs,:,2),2^delta_base{l}(i),coded_base{l}(i)/(length(rs)*1*p*g));
        end
        quants(l).Weights = permute(reshape(quant_vectors(:,:,:,2)*quant_weights,[p,g,q,h,w]),[4,5,1,3,2]);
        wdist_kern{l} = double(sum((quants(l).Weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2));
        non0s_kern{l} = p*g - find([var(quant_weights,[],2);0]==0,1) + 1;
        coded_kern{l} = sum(coded_kern{l},'omitnan');
        coded_base{l} = sum(coded_base{l},'omitnan');
    end
    non0s_kern
    ournet = replaceLayers(neural,quants);

    ydist_kern = cell2mat(ydist_kern);
    coded_kern = cell2mat(coded_kern);
    delta_kern = cell2mat(delta_kern);
    wdist_kern = cell2mat(wdist_kern);
    denom_kern = cell2mat(denom_kern);
    non0s_kern = cell2mat(non0s_kern);

    ydist_base = cell2mat(ydist_base);
    coded_base = cell2mat(coded_base);
    delta_base = cell2mat(delta_base);
    wdist_base = cell2mat(wdist_base);
    denom_base = cell2mat(denom_base);
    
    Y_hats = pred(ournet,images,outlayer);
    Y_cats = getclass(neural,Y_hats);
    
    hist_sum_Y_sse(j,1) = mean((Y_hats(:) - Y(:)).^2,1);
    hist_sum_Y_top(j,1) = mean(images.Labels == Y_cats);
    pred_sum_Y_sse(j,1) = (sum(ydist_kern(:),'omitnan') + sum(ydist_base(:),'omitnan'));
    hist_sum_W_sse(j,1) = (sum(wdist_kern(:),'omitnan') + sum(wdist_base(:),'omitnan'))...
                        / (sum(denom_kern(:),'omitnan'));
    hist_sum_coded(j,1) = (sum(coded_kern(:),'omitnan') + sum(coded_base(:),'omitnan'))...
                        / (sum(denom_kern(:),'omitnan'));
    hist_sum_non0s(j,inlayers) = non0s_kern(:);
    hist_sum_total(j,inlayers) = coded_kern(:) + coded_base(:);
    hist_sum_denom(j,inlayers) = denom_kern(:);

    disp(sprintf('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %4.1f, rate: %5.2e',...
                 archname, tranname, slope, hist_sum_Y_sse(j,1), pred_sum_Y_sse(j,1), ...
                 hist_sum_W_sse(j,1), 100*hist_sum_Y_top(j,1), hist_sum_coded(j,1)));
    if hist_sum_coded(j) == 0  ...
       hist_sum_Y_top(j) <= 0.002
        break;
    end
end

save(sprintf('%s_%s_sum_%d_%d_%d_%s_inter_total',archname,tranname,testsize,inlayers(1),inlayers(end),outlayer),...
     '-v7.3','hist_sum_coded','hist_sum_Y_sse','pred_sum_Y_sse','hist_sum_W_sse','hist_sum_Y_top','hist_sum_non0s',...
     'hist_sum_total','hist_sum_denom');