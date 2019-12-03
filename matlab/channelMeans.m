function [means,offset] = channelMeans(neural,images)
%GENERATE_KL_INTER Generate KL transform for inter-kernel coding.
%   K = GENERATE_KL_INTRA(ARCHNAME,TESTSIZE,KLTTYPE,DIMTYPE)
%   generates the Karhunen-Loeve transform K for neural network
%   architecture ARCHNAME based on TESTSIZE-many images. KLTTYPE
%   can be set to either 'kklt' (default) or 'klt' (not
%   recommended). DIMTYPE can be set to either 1 (separable,
%   default) or 2 (non-separable). 
%
%   Examples:
%   >>  K = generate_KL_intra('alexnet',1000,'kklt',2);
%   produces a non-separable KKLT basis for alexnet using 1000 test
%   images. 
%
%   >>  K = generate_KL_intra('vgg16',2000,'kklt',1);
%   produces a separable KKLT basis for vgg16 using 2000 test
%   images. 
      
    l_kernel = findconv(neural.Layers);
    l_length = length(l_kernel);

    means = cell(l_length,1);
    offset = cell(l_length,1);
    layers = neural.Layers(l_kernel);

    for l = 1:l_length
        layer = layers(l);
        [h,w,p,q,g] = size(layer.Weights);
        X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
        [H,W,P,S] = size(X);
        switch ndims(layer.Weights)
          case 2
            means{l} = mean(X,4);
            offset{l} = layer.Weights*means{l}(:);
          otherwise
            means{l} = mean(mean(mean(X,1),2),4);
            offset{l} = zeros(1,1,q,g);
            for k = 1:g
                offset{l}(1,1,:,k) = squeeze(sum(sum(layer.Weights(:,:,:,:,k),1),2))'*squeeze(means{l}(1,1,(k-1)*p+(1:p)));
            end
        end
    end
end
