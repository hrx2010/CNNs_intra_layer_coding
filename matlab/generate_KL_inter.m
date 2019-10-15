function K = generate_KL_inter(archname,testsize,klttype)
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

    K = cell(l_length,1);
    layers = neural.Layers(l_kernel);

    for l = 1:l_length
        layer = layers(l);
        [h,w,p,q,g] = size(layer.Weights);
        K{l} = cell(1,g);
        X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
        X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means % X = getx(neural,nclass,images,layer.Name);

        for k = 1:g
            % find two KLTs, each using the EVD
            switch klttype
              case 'kklt'
                covX = cov(reshape(permute(X(:,:,(k-1)*p+(1:p),:),[3,1,2,4]),p,[])');
              case 'klt2'
                covX = eye(p);
            end
            invcovX = inv(0.5*(covX+covX'));
            covH = cov(reshape(permute(layer.Weights(:,:,:,:,k),[3,1,2,4]),p,[])');
            [V,~] = eig(0.5*(covH+covH'),0.5*(invcovX+invcovX'));
            K{l}{k} = V';
        end

        disp(sprintf('%s %s | generated transform for layer %03d', archname, klttype, l));
    end        
    save(sprintf('%s_%s_inter',archname,klttype),'K');
end