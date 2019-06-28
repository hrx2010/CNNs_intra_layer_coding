clear all;
close all;

imds = imageDatastore('~/Developer/img','ReadFcn',@vgg16reader);

neural = vgg16;
layers = [neural.Layers(1:39);regressionLayer('Name','output')];
gtruth = predict(assembleNetwork(layers),imds);
normsq = sum(gtruth(:).^2);

l = 2;
[h,w,p,q] = size(layers(l).Weights);

hist_coded = NaN*zeros(16,q);
hist_relse = NaN*zeros(16,q);


for i = 1:1 % iterate over output channels
    quant = layers;
    convw = quant(l).Weights(:,:,:,i);
    biasw = quant(l).Bias(i);
    scale = 2^floor(log2(std(convw(:))/1024));
    coded = Inf;
    for j = 1:16
        if coded == 0
            break
        end
        delta = scale*2^(j-1);
        % quantize each of the q slices
        convq = quantize(convw,delta);
        biasq = quantize(biasw,delta);
        % assemble the network again
        quant(l).Weights(:,:,:,i) = convq;
        quant(l).Bias(i) = biasq;
        net = assembleNetwork(quant);
        % run the prediction
        preds = predict(net,imds);
        relse = sum((preds(:) - gtruth(:)).^2)/normsq;
        coded = qentropy([convq(:);biasq]);

        hist_coded(j,i) = coded;
        hist_relse(j,i) = relse;
        
        disp(sprintf('output %3d, delta: %5.2e, relsse: %5.2e, length: %5.2e',...
                     i, delta, relse, coded));
    end
end
    
