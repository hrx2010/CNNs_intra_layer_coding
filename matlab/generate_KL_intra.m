function K = generate_KL_intra(archname,testsize,klttype,dimtype)
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
        klttype = 'kklt';
    end
    if nargin < 4
        dimtype = 1;
    end

    imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
    labeldir = './ILSVRC2012_val.txt';

    [neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
    [layers,lclass] = removeLastLayer(neural);
    neural = assembleNetwork(layers);
    nclass = assembleNetwork(lclass);

    l_kernel = findconv(neural.Layers); 
    l_length = length(l_kernel);

    K = cell(l_length,1);
    layers = neural.Layers(l_kernel);

    for l = 1:l_length
        layer = layers(l);
        [h,w,p,q,g] = size(layer.Weights);
        K{l} = cell(p,g);
        X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
        X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means % X = getx(neural,nclass,images,layer.Name);

        % find two KLTs, each using the EVD
        for k = 1:g
            for j = 1:p
                covH = covariances(double(layer.Weights(:,:,j,:,k)),dimtype);
                switch klttype
                  case 'kklt'
                    covX = correlation(double(X(:,:,(k-1)*p+j,:)),dimtype,h);
                  case 'klt'
                    covX = eye(h*h);
                end
                invcovX = inv(covX+covX');
                [V,~] = eig(covH+covH',invcovX+invcovX');
                K{l}{j,k} = V';
            end        
        end

        disp(sprintf('%s %s | generated %d-D transform for layer %03d', archname, klttype, dimtype, l));
    end
    save(sprintf('%s_%s%d',archname,klttype,dimtype),'K');
end