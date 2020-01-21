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

trannames = {'idt','dct','dct2_2','klt_100_intra','kkt_100_intra'};
t_length = length(trannames);

gains = zeros(l_length,t_length,2);
layers = neural.Layers(l_kernel);
varH = cell(l_length,length(trannames));
varX = cell(l_length,length(trannames));

for l = 1:2:13%l_length
     layer = layers(l);
     [h,w,p,q,g] = size(layer.Weights);
     X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
     X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means
     H = layer.Weights;

     for t = 1:length(trannames)
         T = gettrans(trannames{t},archname,l);
         varH{l,t} = hvars2(H,T{1},h,w);
         varX{l,t} = xvars2(X,T{3},h,w);
         gains(l,t,1) = geomean(varX{l,t}(:),'omitnan');
         gains(l,t,2) = geomean(varH{l,t}(:),'omitnan');
         disp(sprintf('%s %14s | layer %03d (%5d coefficients) is %5.2f %5.2f %5.2f dB ', ...
                      archname, trannames{t}, l, numel(H), 10*log10(gains(l,t,1)), ...
                      10*log10(gains(l,t,2)), sum(10*log10(gains(l,t,:)))));
     end
end

save(sprintf('%s_gain_intra.mat',archname),'varH','varX')