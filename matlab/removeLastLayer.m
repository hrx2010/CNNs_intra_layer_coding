function [lgraph,lclass] = removeLastLayer(net)
    switch class(net)
      case 'SeriesNetwork'
        lgraph = [net.Layers(1:end-2);regressionLayer('Name','output')];
        lclass = [sequenceInputLayer(net.Layers(end-2).OutputSize,'Name','data');net.Layers(end-1:end)];
      case 'DAGNetwork'
        lgraph = layerGraph(net);
        lgraph = removeLayers(lgraph,{lgraph.Layers(end-1).Name, ...
                              lgraph.Layers(end).Name});
        lgraph = addLayers(lgraph,regressionLayer('Name','output'));
        lgraph = connectLayers(lgraph,lgraph.Layers(end-1).Name, ...
                               'output');
        lclass = [];
    end
end