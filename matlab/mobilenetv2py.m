function net = mobilenetv2py
%load('mobilenetv2.mat','weight','biases','stride','padding','bnorm_mean','bnorm_vars');
    net = mobilenetv2;

    switch class(net)
      case 'SeriesNetwork'
        lgraph = layerGraph(net.Layers);
      case 'DAGNetwork'
        lgraph = layerGraph(net);
    end

    % net = modifyConvParams(lgraph,weight,biases,stride,padding,bnorm_mean,bnorm_vars);
    oldlayer = lgraph.Layers(153);
    newlayer = convolution2dLayer([1,1],1000,'Name','Logits','Weights',permute(oldlayer.Weights,[3,4,2,1]),...
                                  'Bias',permute(oldlayer.Bias,[2,3,1]));
    lgraph = replaceLayer(lgraph,oldlayer.Name,newlayer);
    net = assembleNetwork(lgraph);
end

