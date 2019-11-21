function K = generate_KL_fully(archname,testsize,klttype)
%GENERATE_KL_FULLY Generate KL transform for fully-connected layers.
%   K = GENERATE_KL_FULLY(ARCHNAME,TESTSIZE,KLTTYPE)
%   generates the Karhunen-Loeve transform K for neural network
%   architecture ARCHNAME based on TESTSIZE-many images. KLTTYPE
%   can be set to either 'kklt' (default) or 'klt' (not
%   recommended).
%
%   Examples:
%   >>  K = generate_KL_fully('alexnet',1000,'kklt');
%   produces a KKLT basis for alexnet using 1000 test images. 
%
%   >>  K = generate_KL_fully('vgg16',2000,'kklt');
%   produces a  KKLT basis for vgg16 using 2000 test  images. 



    


    if nargin < 3
        klttype = 'kkt';
    end

    imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
    labeldir = './ILSVRC2012_val.txt';

    [neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
    [layers,lclass] = removeLastLayer(neural);
    neural = assembleNetwork(layers);
    nclass = assembleNetwork(lclass);

    l_kernel = findconv(neural.Layers,{'full'}); 
    l_length = length(l_kernel);

    K = cell(l_length,1);
    layers = neural.Layers(l_kernel);

    for l = 1:l_length
        layer = layers(l);
        [q,p] = size(layer.Weights);
        K{l} = cell(1,1);
        X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
        X = X - mean(X,4); % subtract per-channel means % X = getx(neural,nclass,images,layer.Name);
        
        covH = cov(double(layer.Weights),1);
        % find two KLTs, each using the EVD
        switch klttype
          case 'kkt'
            covX = cov(reshape(double(X),p,[])',1);
          case 'klt'
            covX = eye(p);
        end
        lambdaX = eigs(covX + covX',1) * 1e-6 * eye(p);
        invcovX = inv(covX + covX' + lambdaX);
        [V,D] = eig(covH+covH',invcovX+invcovX','chol');

        [d,I] = sort(diag(D));
        V = V(:,I);
        D = diag(d);
        
        K{l}{1} = V';

        disp(sprintf('%s %s | generated transform for layer %03d/%03d', archname, klttype, l, l_length));
    end        
    save(sprintf('%s_%s_fully',archname,klttype),'K');
end