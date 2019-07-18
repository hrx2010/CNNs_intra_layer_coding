function l = findconv(layers)
    l = [];
    for i = 1:length(layers)
        if isa(layers(i),'nnet.cnn.layer.Convolution2DLayer')
            l = [l,i];
        elseif isa(layers(i),'nnet.cnn.layer.GroupedConvolution2DLayer')
            l = [l,i];
        end
    end
end