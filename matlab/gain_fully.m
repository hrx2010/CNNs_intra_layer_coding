clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
testsize = 4000;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers,{'full'});
l_length = length(l_kernel);

trannames = {'idt_fully','kkt_fully'};
t_length = length(trannames);

gains = zeros(l_length,t_length,2);
layers = neural.Layers(l_kernel);
for l = 1:l_length
     layer = layers(l);
     [q,p] = size(layer.Weights);
     X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
     X = X - mean(X,4); % subtract per-channel means
     H = layer.Weights;

     for t = 1:length(trannames)
         varX = zeros(p,1);
         varH = zeros(p,1);
         T = gettrans(trannames{t},archname,l,1,1);
         varH(:,1) = mean((T{1}*H').^2,2);
         varX(:,1) = mean((T{3}*reshape(double(X),p,[])).^2,2);

         gains(l,t,1) = geomean(varX(:));
         gains(l,t,2) = geomean(varH(:));
         disp(sprintf('%s %s | dense layer %03d/%03d (%8d coefficients) is %7.2f %7.2f %7.2f dB ', ...
                      archname, trannames{t}, l, l_length, numel(H), 10*log10(gains(l,t,1)), ...
                      10*log10(gains(l,t,2)), sum(10*log10(gains(l,t,:)))));
     end
end