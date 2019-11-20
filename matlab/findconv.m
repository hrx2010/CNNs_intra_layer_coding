function l = findconv(layers,types)
    if nargin < 2
        types = {'conv'};
    end
    for i = 1:length(types)
        if ismember('full',types)
            types{end+1} = 'nnet.cnn.layer.FullyConnectedLayer';
        end
        if ismember('conv',types)
            types{end+1} = 'nnet.cnn.layer.Convolution2DLayer';
            types{end+1} = 'nnet.cnn.layer.GroupedConvolution2DLayer';
            types{end+1} = 'nnet.cnn.layer.Convolution2DLayerCustom';
            types{end+1} = 'nnet.cnn.layer.GroupedConvolution2DLayerCustom';
        end
    end

    l = [];
    for i = 1:length(layers)
        if ismember(class(layers(i)),types)
            l = [l,i];
        end
    end
end