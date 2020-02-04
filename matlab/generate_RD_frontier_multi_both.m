function generate_RD_frontier_multi_both(archname,tranname,testsize,inlayers,outlayer,strides)

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

disp(sprintf('%s | top1: %5.2f', archname, 100*mean(images.Labels == Y_cats)));

layers = modifyConvLayers(neural,cmeans,offset);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers); % or specify the layer number directly
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
    slope = -33 + 0.50*(j-1);
    ydist_kern = cell(l_length,1);
    coded_kern = cell(l_length,1);
    delta_kern = cell(l_length,1);
    wdist_kern = cell(l_length,1);
    denom_kern = cell(l_length,1);
    non0s_kern = cell(l_length,1);

    ydist_rows = cell(l_length,1);
    coded_rows = cell(l_length,1);
    delta_rows = cell(l_length,1);
    wdist_rows = cell(l_length,1);
    denom_rows = cell(l_length,1);
    non0s_rows = cell(l_length,1);

    ydist_cols = cell(l_length,1);
    coded_cols = cell(l_length,1);
    delta_cols = cell(l_length,1);
    wdist_cols = cell(l_length,1);
    denom_cols = cell(l_length,1);
    non0s_cols = cell(l_length,1);

    quants = neural.Layers(l_kernel);
    for l = inlayers
        inter_vectors = gettrans([tranname,'_50000_inter'],archname,l);
        intra_vectors = gettrans([tranname,'_50000_intra'],archname,l);
        [h,w,p,q,g] = size(quants(l).Weights);% size(perm5(quants(l).Weights,quants(l),size(quant_vectors,1)));
        quant_weights = quants(l).Weights;
        quant_weights = reshape(inter_vectors(:,:,:,1)*reshape(permute(quant_weights,[3,5,4,1,2]),p*g,q*h*w),p,g,q,h,w);
        quant_weights = reshape(intra_vectors(:,:,:,1)*reshape(permute(quant_weights,[4,5,1,2,3]),h*w,p*g*q),h*w,p*g,q);

        load(sprintf('%s_%s_val_1000_%d_%d_%s_intra_base',archname,tranname,l,l,outlayer));
        [base_best_Y_sse,base_best_delta,base_best_coded] = finddelta(base_Y_sse{l},base_delta{l},base_coded{l});
        ydist_rows{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_Y_sse,2^slope);
        coded_rows{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_coded,2^slope);
        delta_rows{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_delta,2^slope);
        denom_rows{l} = h*w*1*h*w;

        load(sprintf('%s_%s_val_1000_%d_%d_%s_inter_base',archname,tranname,l,l,outlayer));
        [base_best_Y_sse,base_best_delta,base_best_coded] = finddelta(base_Y_sse{l},base_delta{l},base_coded{l});
        ydist_cols{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_Y_sse,2^slope);
        coded_cols{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_coded,2^slope);
        delta_cols{l} = lambda2points(base_best_coded,base_best_Y_sse,base_best_delta,2^slope);
        denom_cols{l} = p*g*1*p*g;

        load(sprintf('%s_%s_val_1000_%d_%d_%s_multi_kern',archname,tranname,l,l,outlayer));
        [kern_best_Y_sse,kern_best_delta,kern_best_coded] = finddelta(kern_Y_sse{l},kern_delta{l},kern_coded{l});
        ydist_kern{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_Y_sse,2^slope);
        coded_kern{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_coded,2^slope);
        delta_kern{l} = lambda2points(kern_best_coded,kern_best_Y_sse,kern_best_delta,2^slope);
        denom_kern{l} = h*w*p*q*g;%ones(size(coded{l}))*(h*w*q);

        [s,t] = deal(strides(l,1),strides(l,2));
        for r = 1:s:h*w
            for c = 1:t:p*g
                rs = r:min(h*w,s+r-1);
                cs = c:min(p*g,t+c-1);
                scale = floor(log2(sqrt(mean(reshape(quant_weights(rs,cs,:),[],1).^2))));
                if scale < -24 %all zeros
                    continue
                end
                scale = floor(log2(sqrt(mean(reshape(quant_weights(rs,cs,:),[],1).^2))));
                % quantize for the given lambda
                quant_weights(rs,cs, :) = quantize(quant_weights(rs,cs, :),2^delta_kern{l}(r,c),coded_kern{l}(r,c)/(s*t*q));
                intra_vectors(:,rs,:,2) = quantize(intra_vectors(:,rs,:,2),2^delta_rows{l}(r,1),coded_rows{l}(r,1)/(s*h*w));
                inter_vectors(:,cs,:,2) = quantize(inter_vectors(:,cs,:,2),2^delta_cols{l}(c,1),coded_cols{l}(c,1)/(t*p*g));
            end
        end
        
        quant_weights = permute(reshape(intra_vectors(:,:,:,2)*reshape(quant_weights,h*w,p*g*q),h,w,p,g,q),[3,4,5,1,2]);
        quant_weights = permute(reshape(inter_vectors(:,:,:,2)*reshape(quant_weights,p*g,q*h*w),p,g,q,h,w),[4,5,1,3,2]);
        quants(l).Weights = quant_weights;

        wdist_kern{l} = double(sum((quant_weights(:) - neural.Layers(l_kernel(l)).Weights(:)).^2));
        non0s_kern{l} = sum(squeeze(max(max(max(abs(quant_weights),[],4),[],1),[],2))>1e-7);
        ydist_kern{l} = sum(ydist_kern{l}(:),'omitnan');
        coded_kern{l} = sum(coded_kern{l}(:),'omitnan');
        coded_rows{l} = sum(coded_rows{l}(:),'omitnan');
        coded_cols{l} = sum(coded_cols{l}(:),'omitnan');
    end
    ournet = replaceLayers(neural,quants);

    ydist_kern = cell2mat(ydist_kern);
    coded_kern = cell2mat(coded_kern);
    % delta_kern = cell2mat(delta_kern);
    wdist_kern = cell2mat(wdist_kern);
    denom_kern = cell2mat(denom_kern);
    non0s_kern = cell2mat(non0s_kern);

    ydist_rows = cell2mat(ydist_rows);
    coded_rows = cell2mat(coded_rows);
    % delta_rows = cell2mat(delta_rows);
    wdist_rows = cell2mat(wdist_rows);
    denom_rows = cell2mat(denom_rows);

    ydist_cols = cell2mat(ydist_cols);
    coded_cols = cell2mat(coded_cols);
    % delta_cols = cell2mat(delta_cols);
    wdist_cols = cell2mat(wdist_cols);
    denom_cols = cell2mat(denom_cols);
    
    Y_hats = pred(ournet,images,outlayer);
    Y_cats = getclass(neural,Y_hats);
    
    hist_sum_Y_sse(j,1) = mean((Y_hats(:) - Y(:)).^2,1);
    hist_sum_Y_top(j,1) = mean(images.Labels == Y_cats);
    pred_sum_Y_sse(j,1) = (sum(ydist_kern(:),'omitnan') + sum(ydist_rows(:),'omitnan'));
    hist_sum_W_sse(j,1) = (sum(wdist_kern(:),'omitnan') + sum(wdist_rows(:),'omitnan') + sum(wdist_cols(:),'omitnan'))...
                        / (sum(denom_kern(:),'omitnan'));
    hist_sum_coded(j,1) = (sum(coded_kern(:),'omitnan') + sum(coded_rows(:),'omitnan') + sum(coded_cols(:),'omitnan'))...
                        / (sum(denom_kern(:),'omitnan'));
    hist_sum_non0s(j,inlayers) = non0s_kern(:);
    hist_sum_total(j,inlayers) = coded_kern(:) + coded_rows(:) + coded_cols(:);
    hist_sum_denom(j,inlayers) = denom_kern(:);

    disp(sprintf('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %5.2f, rate: %5.2e',...
                 archname, tranname, slope, hist_sum_Y_sse(j,1), pred_sum_Y_sse(j,1), ...
                 hist_sum_W_sse(j,1), 100*hist_sum_Y_top(j,1), hist_sum_coded(j,1)));
    if hist_sum_coded(j) == 0 || ...
       hist_sum_Y_top(j) <= 0.002
        break;
    end
end

save(sprintf('%s_%s_sum_%d_%d_%d_%s_multi_total',archname,tranname,testsize,inlayers(1),inlayers(end),outlayer),...
     'hist_sum_coded','hist_sum_Y_sse','pred_sum_Y_sse','hist_sum_W_sse','hist_sum_Y_top','hist_sum_non0s',...
     'hist_sum_total','hist_sum_denom');