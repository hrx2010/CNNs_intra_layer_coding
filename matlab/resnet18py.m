function net = resnet18py
    load('resnet18.mat','weight','biases','stride','padding','bnorm_mean','bnorm_vars');
    net = resnet18;

    switch class(net)
      case 'SeriesNetwork'
        lgraph = layerGraph(net.Layers);
      case 'DAGNetwork'
        lgraph = layerGraph(net);
    end
    lgraph = modifyConvParams(lgraph,weight,biases,stride,padding,bnorm_mean,bnorm_vars);
    oldlayer = lgraph.Layers(2);
    lgraph = removeLayers(lgraph,oldlayer.Name);
    lgraph = connectLayers(lgraph,'data','conv1');
    oldlayer = lgraph.Layers(5);
    % newlayer = maxPooling2dLayer(oldlayer.PoolSize,'Stride',oldlayer.Stride,'Padding',1,'Name',oldlayer.Name);
    % lgraph = replaceLayer(lgraph,oldlayer.Name,newlayer);
    net = assembleNetwork(lgraph);
end