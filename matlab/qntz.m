clear all;
close all;

imds = imageDatastore('~/Developer/img','ReadFcn',@vgg16reader);

neural = vgg16;
layers = [neural.Layers(1:39);regressionLayer('Name','output')];
gtruth = predict(assembleNetwork(layers),imds);

l = 2;
[h,w,p,q] = size(layers(l).Weights);

for i = 1:q
    quant = layers;
    convw = quant(l).Weights(:,:,:,i);
    biasw = quant(l).Bias(i);
    scale = std(convw(:));
    coded = Inf;

    delta = 2^floor(log2(scale/256));
    while (coded > 0)
        % quantize each of the q slices
        convw = quantize(convw,delta); 
        biasw = quantize(biasw,delta);
        coded = qentropy([convw(:);biasw]);
        % assemble the network again
        quant(l).Weights(:,:,:,i) = convw;
        quant(l).Bias(i) = biasw;
        net = assembleNetwork(quant);
        % run the prediction
        mse = mean(sum((predict(net,imds) - gtruth).^2,2),1);
        delta = 2 * delta;
    end
end
    
