function lgraph =  modifyConvParams(lgraph,weight,biases,bnorm_mean,bnorm_vars)
    if nargin < 4
        bnorm_mean = cell(length(weight),1);
        bnorm_vars = cell(length(weight),1);
    end

    oldlayer = lgraph.Layers(1);
    newlayer = imageInputLayer(oldlayer.InputSize,'Normalization','none','Name',oldlayer.Name);
    lgraph = replaceLayer(lgraph,oldlayer.Name,newlayer);

    inlayers = lgraph.Layers;
    l_kernel = findconv(inlayers);
    l_length = length(l_kernel);
    
    for l = 1:l_length
        disp(l);
        oldlayer = inlayers(l_kernel(l));
        newlayer = oldlayer;
        switch class(oldlayer)
          case 'nnet.cnn.layer.Convolution2DLayer'
            [h,w,p,q,g] = size(oldlayer.Weights);
            newlayer.Weights = reshape(permute(weight{l},[3,4,2,1]),[h,w,p,q,g]);
            newlayer.Bias = reshape(permute(biases{l},[2,3,1]),[1,1,q,g]);
          case 'nnet.cnn.layer.GroupedConvolution2DLayer'
            [h,w,p,q,g] = size(oldlayer.Weights);
            newlayer.Weights = reshape(permute(weight{l},[3,4,2,1]),[h,w,p,q,g]);
            newlayer.Bias(:) = reshape(permute(biases{l},[2,3,1]),[1,1,q,g]);
          case 'nnet.cnn.layer.FullyConnectedLayer'
            [q,p] = size(oldlayer.Weights);
            newlayer.Weights = permute(weight{l},[1,2,3,4]);
            newlayer.Bias(:) = reshape(permute(biases{l},[1,2,3]),[q,1]);
          case 'nnet.cnn.layer.BatchNormalizationLayer'
            newlayer.Scale = permute(weight{l},[3,1,2]);
            newlayer.Offset = permute(biases{l},[3,1,2]);
            newlayer.TrainedMean = permute(bnorm_mean{l},[3,1,2]);
            newlayer.TrainedVariance = permute(bnorm_vars{l},[3,1,2]);
            newlayer.Epsilon = 1e-5;
        end
        lgraph = replaceLayer(lgraph,oldlayer.Name,newlayer);
    end

end

