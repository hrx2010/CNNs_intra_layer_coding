function generate_RD_frontier(archname,tranname,testsize,inlayers,outlayer)

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
% archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';

maxsteps = 96;

load(sprintf('%s_%s_val_1000_%s',archname,tranname,outlayer));
[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);
nclass = assembleNetwork(lclass);

l_kernel = findconv(neural.Layers); % or specify the layer number directly
l_length = length(l_kernel);

hist_sum_Y_top = zeros(maxsteps,1,testsize)*NaN;
hist_sum_Y_sse = zeros(maxsteps,1,testsize)*NaN;
pred_sum_Y_sse = zeros(maxsteps,1,testsize)*NaN;
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
        K = gettrans(tranname,archname,l);
        quants(l).Weights = transform(quants(l).Weights,K{1});
        [h,w,p,q] = size(quants(l).Weights);

        [best_Y_sse,best_delta,best_coded] = finddelta(mean(hist_Y_sse{l},4),hist_delta{l},hist_coded{l});
        ydist{l} = lambda2points(best_coded,best_Y_sse,best_Y_sse,2^slope);
        coded{l} = lambda2points(best_coded,best_Y_sse,best_coded,2^slope);
        delta{l} = lambda2points(best_coded,best_Y_sse,best_delta,2^slope);
        denom{l} = ones(size(coded{l}))*(p*q);
        for i = 1:h*w
            [r,c] = ind2sub([h,w],i);
            % quantize for the given lambda
            quants(l).Weights(r,c,:) = quantize(quants(l).Weights(r,c,:),2^delta{l}(i),coded{l}(i)/(p*q));
            %assert(qentropy(quants(l).Weights(r,c,:))*(p*q) == coded{l}(i));
        end
        quants(l).Weights = transform(quants(l).Weights,K{2});
        wdist{l} = double(sum(reshape(quants(l).Weights - neural.Layers(l_kernel(l)).Weights,h*w,[]).^2,2));
    end
    ournet = replaceLayers(neural,quants);

    ydist = cell2mat(ydist);
    coded = cell2mat(coded);
    delta = cell2mat(delta);
    wdist = cell2mat(wdist);
    denom = cell2mat(denom);
    
    [Y_hats,Y_cats] = pred(ournet,nclass,images,outlayer);
    hist_sum_Y_sse(j,1,:) = mean((Y_hats - Y).^2,1);
    hist_sum_Y_top(j,1,:) = images.Labels == Y_cats;
    pred_sum_Y_sse(j,1,:) = sum(ydist,1);
    hist_sum_W_sse(j,1,1) = sum(wdist(:))/sum(denom(:));
    hist_sum_coded(j,1,1) = sum(coded(:))/sum(denom(:));

    disp(sprintf('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %4.1f, rate: %5.2e',...
         archname, tranname, slope, mean(hist_sum_Y_sse(j,1,:)), mean(pred_sum_Y_sse(j,1,:)), ...
         mean(hist_sum_W_sse(j,1)), 100*mean(hist_sum_Y_top(j,1,:)), mean(hist_sum_coded(j,1))));
    if hist_sum_coded(j) == 0
        break;
    end
end

save(sprintf('%s_%s_sum_%d_%d_%d_%s',archname,tranname,testsize,inlayers(1),inlayers(end),outlayer),...
     'hist_sum_coded','hist_sum_Y_sse','pred_sum_Y_sse','hist_sum_W_sse','hist_sum_Y_top');