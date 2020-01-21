clear all;
close all;

archname = 'vgg16';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
testsize = 100;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers);
l_length = length(l_kernel);

trannames = {'idt_100_inter','klt_100_inter','kkt_100_inter'};
t_length = length(trannames);

gains = zeros(l_length,t_length,2);
layers = neural.Layers(l_kernel);
varH = cell(l_length,length(trannames));
varX = cell(l_length,length(trannames));

for l = [1:2:13,14:16]%l_length
     layer = layers(l);
     [h,w,p,q,g] = size(perm5(layer.Weights,layer));
     X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
     X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means
     H = layer.Weights;

     for t = 1:length(trannames)
         T = gettrans(trannames{t},archname,l);
         varH{l,t} = mean(reshape(permute(reshape5(transform_inter(perm5(H,layer),T{1}),T{1}),[3,5,1,2,4]),p,g,[]).^2,3);
         varX{l,t} = mean(reshape(permute(reshape5(transform_inter(X,T{3}),T{3}),[3,5,1,2,4]),p,g,[]).^2,3);
         gains(l,t,1) = geomean(varX{l,t}(:));
         gains(l,t,2) = geomean(varH{l,t}(:));
         disp(sprintf('%s %s | layer %03d (%5d coefficients) is %5.2f %5.2f %5.2f dB ', ...
                      archname, trannames{t}, l, numel(H), 10*log10(gains(l,t,1)), ...
                      10*log10(gains(l,t,2)), sum(10*log10(gains(l,t,:)))));
     end
end

save(sprintf('%s_gain_inter.mat',archname),'varH','varX')