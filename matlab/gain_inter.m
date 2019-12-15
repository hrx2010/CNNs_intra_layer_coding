clear all;
close all;

archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
testsize = 100;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers);
l_length = length(l_kernel);

trannames = {'idt_5000_inter','klt_5000_inter','kkt_5000_inter'};
t_length = length(trannames);

gains = zeros(l_length,t_length,2);
layers = neural.Layers(l_kernel);
for l = 1:5%l_length
     layer = layers(l);
     [h,w,p,q,g] = size(layer.Weights);
     X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
     X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means
     H = layer.Weights;

     for t = 1:length(trannames)
         T = gettrans(trannames{t},archname,l);
         varH = mean(reshape(permute(transform_inter(H,T{1}),[3,5,1,2,4]),p,g,[]).^2,3);
         varX = mean(reshape(permute(transform_inter(X,T{3}),[3,5,1,2,4]),p,g,[]).^2,3);
         gains(l,t,1) = geomean(varX(:));
         gains(l,t,2) = geomean(varH(:));
         disp(sprintf('%s %s | layer %03d (%5d coefficients) is %5.2f %5.2f %5.2f dB ', ...
                      archname, trannames{t}, l, numel(H), 10*log10(gains(l,t,1)), ...
                      10*log10(gains(l,t,2)), sum(10*log10(gains(l,t,:)))));
     end
end