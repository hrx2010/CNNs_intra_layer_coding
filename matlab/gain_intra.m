clear all;
close all;

archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
testsize = 1000;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers);
l_length = length(l_kernel);

trannames = {'kklt1'};
t_length = length(trannames);

gains = zeros(l_length,t_length,2);
layers = neural.Layers(l_kernel);
for l = 1:l_length
     layer = layers(l);
     [h,w,p,q,g] = size(layer.Weights);
     X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
     X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means
     H = layer.Weights;

     for t = 1:length(trannames)
         varX = zeros(h*w,p*g);
         varH = zeros(h*w,p*g);
         for k = 1:g
             for j = 1:p
                 T = gettrans(trannames{t},archname,l,j,k);
                 % H = layer.Weights(:,:,j,:,k);
                 % covH = covariances(H(:,:,j,:,k),2);
                 % covX = correlation(X(:,:,(k-1)*p+j,:),2,h);
                 % varH(:,(k-1)*p+j) = diag(blktrans(blktrans(covH,T{1},h,w)',T{1},w,h)');
                 varH(:,(k-1)*p+j) = reshape(hvars2(H(:,:,j,:,k),T{1},h,w),[],1);
                 % varX(:,(k-1)*p+j) = diag(blktrans(blktrans(covX,T{3},h,w)',T{3},w,h)');
                 varX(:,(k-1)*p+j) = reshape(xvars2(X(:,:,(k-1)*p+j,:),T{3},h,w),[],1);
             end
         end
         % varX(varX(:) < 0) = NaN;
         % varH(varH(:) < 0) = NaN;
         gains(l,t,1) = geomean(varX(:),'omitnan');
         gains(l,t,2) = geomean(varH(:),'omitnan');
         disp(sprintf('%s %s | layer %03d (%5d coefficients) is %5.2f %5.2f %5.2f dB ', ...
                      archname, trannames{t}, l, numel(H), 10*log10(gains(l,t,1)), ...
                      10*log10(gains(l,t,2)), sum(10*log10(gains(l,t,:)))));
     end
end