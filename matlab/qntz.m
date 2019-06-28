clear all;
close all;

imds = imageDatastore('~/Developer/ILSVRC2012/ILSVRC2012_test_00000*.JPEG','ReadFcn',@vgg16reader);

neural = vgg16;
layers = [neural.Layers(1:39);regressionLayer('Name','output')];
neural = assembleNetwork(layers);

l = 2; %layer to get the RD curves for
[h,w,p,q] = size(layers(l).Weights);

steps = 32;
hist_coded = zeros(steps,q)*NaN;
hist_relse = zeros(steps,q);
norm2 = 0;

for f = 1:length(imds.Files)
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
            error = sum((Y_hat(:) - Y(:)).^2);
            coded = qentropy([convq(:);biasq]);
            
            hist_coded(j,i) = coded;
            hist_relse(j,i) = hist_relse(j,i) + error;
            
            [~,filename] = fileparts(imds.Files{f});
            disp(sprintf('%s, output %3d, delta: %5.2e, relsse: %5.2e, rate: %5.2e',...
                         filename, i, delta, sqrt(error/Y_ssq), coded));
        end
    end
end    
