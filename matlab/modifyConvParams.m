function lgraph =  modifyConvParams(neural,weight,biases)
    switch class(neural)
      case 'SeriesNetwork'
        lgraph = layerGraph(neural.Layers);
      case 'DAGNetwork'
        lgraph = layerGraph(neural);
    end

    oldlayer = neural.Layers(1);
    newlayer = imageInputLayer(oldlayer.InputSize,'Normalization','none','Name',oldlayer.Name);
    lgraph = replaceLayer(lgraph,oldlayer.Name,newlayer);

    inlayers = lgraph.Layers;
    l_kernel = findconv(inlayers);
    l_length = length(l_kernel);
    
    for l = 1:l_length
        oldlayer = inlayers(l_kernel(l));
        newlayer = oldlayer;
        switch class(oldlayer)
          case 'nnet.cnn.layer.Convolution2DLayer'
            newlayer.Weights = permute(weight{l},[3,4,2,1]);
            newlayer.Bias = permute(biases{l},[3,1,2]);
          case 'nnet.cnn.layer.FullyConnectedLayer'
            newlayer.Weights = permute(weight{l},[1,2,3,4]);
            newlayer.Bias = permute(biases{l},[2,1,3]);
        end
        lgraph = replaceLayer(lgraph,oldlayer.Name,newlayer);
    end

    lgraph = assembleNetwork(lgraph);
end

