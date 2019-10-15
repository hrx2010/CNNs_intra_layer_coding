clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
testsize = 1000;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers);
l_length = length(l_kernel);

trannames = {'idt2','dst2','dct2','klt2','kklt'};
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
         varX = zeros(h,w,p*g);
         varH = zeros(h,w,p*g);
         for k = 1:g
             for j = 1:p
                 trans = gettrans(trannames{t},archname,l,j,k);
                 i = (k-1)*p+j;
                 varW(:,:,i) = mean(xvars2(X(:,:,i,:),size(H,1),size(H,2),trans{3}),4);
                 varH(:,:,i) = mean(abs(transform(H(:,:,j,:,k),trans{1})).^2,4);
             end
         end
         gains(l,t,1) = geomean(varW(:));
         gains(l,t,2) = geomean(varH(:));
         disp(sprintf('%s %s | layer %03d (%5d coefficients) is %5.2f %5.2f %5.2f dB ', ...
                      archname, trannames{t}, l, numel(H), 10*log10(gains(l,t,1)), ...
                      10*log10(gains(l,t,2)), sum(10*log10(gains(l,t,:)))));
     end
end