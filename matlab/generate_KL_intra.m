function T = generate_KL_intra(archname,testsize,klttype)
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
      

    
    if nargin < 3
        klttype = 'kkt';
    end

    imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
    labeldir = './ILSVRC2012_val.txt';

    [neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
    [layers,lclass] = removeLastLayer(neural);
    neural = assembleNetwork(layers);
    nclass = assembleNetwork(lclass);

    l_kernel = findconv(neural.Layers); 
    l_length = length(l_kernel);

    T = cell(l_length,1);
    layers = neural.Layers(l_kernel);

    for l = 1:l_length
        layer = layers(l);
        layer_weights = perm5(layer.Weights,layer);
        [h,w,p,q,g] = size(layer_weights);
        T{l} = zeros(h*w,h*w,p*g,2);
        X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);

        switch ndims(layer.Weights)
          case 2
            X = reshape(X-mean(X,4),1,1,p,[]);
          otherwise
            X = X - mean(mean(mean(X,1),2),4);
        end
        % find two KLTs, each using the EVD
        for k = 1:g
            for j = 1:p
                switch klttype
                  case 'kkt'
                    covH = covariances(double(layer_weights(:,:,j,:,k)),1);
                    covX = correlation(double(X(:,:,(k-1)*p+j,:)),1,h);
                  case 'klt'
                    covH = covariances(double(layer_weights(:,:,j,:,k)),1);
                    covX = eye(h*w);
                  case 'idt'
                    covH = eye(h*w);
                    covX = eye(h*w);
                end
                invcovX = inv(covX+covX'+0.01*eye(h*w)*eigs(covX+covX',1));
                [V,d] = eig(covH+covH',invcovX+invcovX','chol','vector');
                invVt = inv(V')./sqrt(sum(inv(V').^2));
                T{l}(:,:,(k-1)*p+j,1) = inv(invVt(:,end:-1:1));
                T{l}(:,:,(k-1)*p+j,2) = invVt(:,end:-1:1);
            end        
        end

        disp(sprintf('%s %s | generated intra transform for layer %03d using %d images',...
                     archname, klttype, l, testsize));
    end
    save(sprintf('%s_%s_%d_intra',archname,klttype,testsize),'-v7.3','T');
end