clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
testsize = 1;
maxsteps = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

layers(l).Weights = fft2split(fftshift(fftshift(fft2(neural.Layers(l).Weights),1),2));

% search for the best delta
N = 10;
M = 256;
mses = Inf*ones(M,N);
levs = 2.^((1:N)+1);
for n = 1:N
    c = -levs(n)/2:levs(n)/2-1; %canonical lattice
    b = 0.5*(c(1:end-1)+c(2:end)); %midpoints
    deltas = logspace(-3,2,M);
    for m = 1:length(deltas)
        delta = deltas(m);
        [~,~,mses(m,n)] = quantiz(layers(l).Weights(:),delta*b,delta*c);
    end
end
[~,idx] = min(mses);

figure(1);
loglog(levs,deltas(idx),'.-');
axis([10^0,10^4,10^-2,10^1]);
xlabel('Reconstruction levels');
ylabel('Optimal step-size');
pdfprint('temp1.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);

figure(2);
loglog(deltas,mses);
axis([10^-3,10^2,10^-5,10^0]);
xlabel('Quantization step-size');
ylabel('Mean squared-error');
pdfprint('temp2.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);
