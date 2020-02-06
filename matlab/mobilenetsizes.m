function strides = mobilenetsizes

strides = [];
net = mobilenetv2py;
l_layers = findconv(net.Layers,{'conv'});
for i = l_layers
    if strcmp(class(net.Layers(i)),'nnet.cnn.layer.GroupedConvolution2DLayer')
        strides = [strides;size(net.Layers(i).Weights,5)];
    elseif strcmp(class(net.Layers(i)),'nnet.cnn.layer.Convolution2DLayer')
        strides = [strides;size(net.Layers(i).Weights,3)];
    end
end

strides(1) = 1;
strides(2:end) = strides(2:end)/8;