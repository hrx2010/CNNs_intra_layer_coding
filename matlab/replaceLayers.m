function net = replaceLayers(net,layers)
    switch class(net)
      case 'SeriesNetwork'
        lgraph = layerGraph(net.Layers);
      case 'DAGNetwork'
        lgraph = layerGraph(net);
    end
    for l = 1:length(layers)
        lgraph = replaceLayer(lgraph,layers(l).Name,layers(l));
    end
    net = assembleNetwork(lgraph);
end