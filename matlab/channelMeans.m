function [means,offset] = channelMeans(archname, testsize)
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

    imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
    labeldir = './ILSVRC2012_val.txt';
    [neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
      
    l_kernel = findconv(neural.Layers);
    l_length = length(l_kernel);

    means = cell(l_length,1);
    offset = cell(l_length,1);
    layers = neural.Layers(l_kernel);

    for l = 14:l_length
        layer = layers(l);
        layer_weights = perm5(layer.Weights,layer);
        [h,w,p,q,g] = size(layer_weights);
        X_mean = predmean(neural,images,neural.Layers(l_kernel(l)-1).Name,p,g);

        if p ~= size(X_mean,1)
            X_mean = reshape(permute(repmat(X_mean,[1,1,1,p/size(X_mean,3)]),[1,2,4,3]),1,1,[]);
        end

        means{l} = X_mean;
        offset{l} = zeros(1,1,q,g);
        for k = 1:g
            offset{l}(1,1,:,k) = squeeze(sum(sum(layer_weights(:,:,:,:,k),1),2))'*squeeze(means{l}(1,1,(k-1)*p+(1:p)));
        end
    end
end
