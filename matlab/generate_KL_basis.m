function K = generate_KL_basis(archname,testsize,klttype)
    if nargin < 3
        klttype = 'kklt';
    end

    imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
    labeldir = './ILSVRC2012_val.txt';

    [neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
    [layers,lclass] = removeLastLayer(neural);
    neural = assembleNetwork(layers);
    nclass = assembleNetwork(lclass);

    l_kernel = findconv(neural.Layers); 
    l_length = length(l_kernel);

    K = cell(l_length,2);
    layers = neural.Layers(l_kernel);

    for l = 1:l_length
        layer = layers(l);
        [h,w,p,q] = size(layer.Weights);
        switch klttype
          case 'kklt'
            X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
            X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means % X = getx(neural,nclass,images,layer.Name);
            covX = blktoeplitz(autocorr3(X,h,w));
          case 'klt'
            covX = eye(h*w);
        end

        invcovX = inv(covX+covX');
        covH = cov(reshape(layer.Weights,[h*w,p*q])');
        [V,~] = eig(covH+covH',invcovX+invcovX');
        K{l,2} = V';
        K{l,1} = inv(V');
        disp(sprintf('%s %s | generated transform for layer %03d', archname, klttype, l));
    end
    save(sprintf('%s_%s',archname,klttype),'K');
end