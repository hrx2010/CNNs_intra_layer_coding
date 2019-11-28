function K = generate_KL_inter(archname,testsize,klttype)
%GENERATE_KL_INTER Generate KL transform for intra-kernel coding.
%   K = GENERATE_KL_INTRA(ARCHNAME,TESTSIZE,KLTTYPE)
%   generates the Karhunen-Loeve transform K for neural network
%   architecture ARCHNAME based on TESTSIZE-many images. KLTTYPE
%   can be set to either 'kklt' (default) or 'klt' (not
%   recommended).
%
%   Examples:
%   >>  K = generate_KL_inter('alexnet',1000,'kklt');
%   produces a KKLT basis for alexnet using 1000 test images. 
%
%   >>  K = generate_KL_inter('vgg16',2000,'kklt');
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

    l_kernel = findconv(neural.Layers); 
    l_length = length(l_kernel);

    T = cell(l_length,1);
    layers = neural.Layers(l_kernel);

    for l = 1:l_length
        layer = layers(l);
        [h,w,p,q,g] = size(layer.Weights);
        T{l} = zeros(p,p,g,4);
        X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
        X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means % X = getx(neural,nclass,images,layer.Name);

        for k = 1:g
            for j = 1:1 % only one transform per group
                covH = cov(reshape(permute(double(layer.Weights(:,:,:,:,k)),[3,1,2,4]),p,[])',1);
                switch klttype
                  case 'kkt'
                    covX = cov(reshape(permute(double(X(:,:,(k-1)*p+(1:p),:)),[3,1,2,4]),p,[])',1);
                  case 'klt'
                    covX = eye(p);
                end
                invcovX = inv(covX+covX');
                [V,~] = eig(covH+covH',invcovX+invcovX','chol');
                invVt = inv(V')./sqrt(sum(inv(V').^2));
                T{l}(:,:,k,1) = inv(invVt);
                T{l}(:,:,k,2) = invVt;
                T{l}(:,:,k,3) = T{l}(:,:,k,1)';
                T{l}(:,:,k,4) = T{l}(:,:,k,2)';
            end
        end
        T{l} = reshape(T{l},[p,p*g,1,4]);
        disp(sprintf('%s %s | generated transform for layer %03d', archname, klttype, l));
    end        
    save(sprintf('%s_%s_inter',archname,klttype),'T');
end