function l = findconv(layers)
    for i = 1:length(layers)
        if isa(layers(i),'nnet.cnn.layer.Convolution2DLayer')
            l = i;
            break;
        end
    end
end