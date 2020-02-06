function net = resnet50py
    load('resnet50.mat','weight','biases','stride','padding','bnorm_mean','bnorm_vars');
    net = resnet50;

    switch class(net)
      case 'SeriesNetwork'
        lgraph = layerGraph(net.Layers);
      case 'DAGNetwork'
        lgraph = layerGraph(net);
    end
    lgraph = modifyConvParams(lgraph,weight,biases,stride,padding,bnorm_mean,bnorm_vars);
    oldlayer = lgraph.Layers(5);
    newlayer = maxPooling2dLayer(oldlayer.PoolSize,'Stride',oldlayer.Stride,'Padding',1,'Name',oldlayer.Name);
    lgraph = replaceLayer(lgraph,oldlayer.Name,newlayer);
    net = assembleNetwork(lgraph);
end