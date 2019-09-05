function K = generate_KL_basis(archname,testsize)
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
        X = getx(neural,nclass,images,layer.Name);
        covX = blktoeplitz(autocorr3(X,h,w));
        invcovX = inv(covX);
        covH = cov(reshape(layer.Weights,[h*w,p*q])');
        K{l,1} = eig(covH,(invcovX+invcovX')/2);
        K{l,2} = inv(K{l,1});
    end
end