clear all;
close all;

% Choose from: 'alexnet', 'vgg16' and 'resnet50'
arch = 'vgg16;' 
% Specify the filepath to ILSVRC test images
filepath = '~/Developer/ILSVRC2012/ILSVRC2012_test_00000*.JPEG';
% Number of test files to perform prediction. Set to 0 to test all
% files (not recommended)
testsize = 32;

switch arch
  case 'alexnet'
    readerfun = @alexnetreader;
    neural = alexnet;
  case 'vgg16'
    readerfun = @vgg16reader;
    neural = vgg16;
  case 'resnet50'
    readerfun = @vgg16reader;
    neural = resnet50;
end

imds = imageDatastore(filepath,'ReadFcn',readerfun);
layers = [neural.Layers(1:end-2);regressionLayer('Name','output')];
neural = assembleNetwork(layers);

l = 2; %layer to get the RD curves for
[h,w,p,q] = size(layers(l).Weights);

steps = 32;
hist_coded = zeros(testsize,steps,q)*NaN;
hist_Y_sse = zeros(testsize,steps,q)*NaN;
norm2 = 0;

for f = 1:testsize%
    X = imds.readimage(f);
    Y = predict(neural,X);
    Y_ssq = sum(Y(:).^2);
    norm2 = norm2 + Y_ssq;
    for i = 1:q % iterate over output channels
        quant = layers;
        convw = quant(l).Weights(:,:,:,i);
        biasw = quant(l).Bias(i);
        scale = 2^floor(log2(std(convw(:))/1024));
        coded = Inf;
        for j = 1:steps
            if coded == 0
                break
            end
            delta = scale*sqrt(2^(j-1));
            % quantize each of the q slices
            convq = quantize(convw,delta);
            biasq = quantize(biasw,delta);
            % assemble the network again
            quant(l).Weights(:,:,:,i) = convq;
            quant(l).Bias(i) = biasq;
            net = assembleNetwork(quant);
            % run the prediction
            Y_hat = predict(net,X);
            Y_sse = sum((Y_hat(:) - Y(:)).^2);
            coded = qentropy([convq(:);biasq]);
            
            hist_coded(f,j,i) = coded;
            hist_Y_sse(f,j,i) = Y_sse;
            
            [~,filename] = fileparts(imds.Files{f});
            disp(sprintf('%s | slice %03d, delta: %5.2e, relerr: %5.2e, rate: %5.2e',...
                         filename, i, delta, sqrt(Y_sse/Y_ssq), coded));
        end
    end
end    
