function sz = mobilenetsizes

sz = [];
net = mobilenetv2;
l_layers = findconv(net.Layers,{'conv'});
for i = l_layers
    if strcmp(class(net.Layers(i)),'nnet.cnn.layer.GroupedConvolution2DLayer')
        sz = [sz;size(net.Layers(i).Weights,5)];
    elseif strcmp(class(net.Layers(i)),'nnet.cnn.layer.Convolution2DLayer')
        sz = [sz;size(net.Layers(i).Weights,3)];
    end
end