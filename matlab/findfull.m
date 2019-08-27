function l = findfull(layers)
    l = [];
    for i = 1:length(layers)
        if isa(layers(i),'nnet.cnn.layer.FullyConnectedLayer')
            l = [l,i];
        end
    end
end