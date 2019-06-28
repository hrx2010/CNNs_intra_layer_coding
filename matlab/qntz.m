clear all;
close all;

% Choose from: 'alexnet', 'vgg16', 'resnet50', and 'mobilenetv2', and
% specify the filepath to ILSVRC test images. Number of test files to
% predict can be set manually or set to 0 to predict all files in the
% datastore (not recommended)
archname = 'vgg16';
filepath = '~/Developer/ILSVRC2012/ILSVRC2012_test_00000*.JPEG';
testsize = 32;

switch archname
  case 'alexnet'
    readerfun = @read227x227;
    neural = alexnet;
  case 'vgg16'
    readerfun = @read224x224;
    neural = vgg16;
  case 'resnet50'
    readerfun = @read224x224;
    neural = resnet50;
  case 'mobilenetv2'
    readerfun = @read224x224;
    neural = mobilenetv2;
end

imds = imageDatastore(filepath,'ReadFcn',readerfun);
layers = [neural.Layers(1:end-2);regressionLayer('Name','output')];
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

steps = 32;
hist_coded = zeros(testsize,steps,q)*NaN;
hist_Y_sse = zeros(testsize,steps,q)*NaN;

parfor f = 1:testsize%
    X = imds.readimage(f);
    Y = predict(neural,X);
    Y_ssq = sum(Y(:).^2);
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
            % assemble the net using layers
            quant(l).Weights(:,:,:,i) = convq;
            quant(l).Bias(i) = biasq;
            net = assembleNetwork(quant);
            % run the prediction on image X
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
