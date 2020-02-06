function T = generate_KL_inter(archname,testsize,klttype)
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

    l_kernel = findconv(neural.Layers,{'conv'});
    l_length = length(l_kernel);

    T = cell(l_length,1);
    layers = neural.Layers(l_kernel);

    for l = 1:l_length
        layer = layers(l);
        layer_weights = layer.Weights;
        [X_mean, X_vars] = predmean(neural,images,neural.Layers(l_kernel(l)-1).Name,size(layer_weights,5),size(layer_weights,3));
        [h,w,p,q,g] = size(layer_weights);
        T{l} = zeros(p*g,p*g,1,2);

        for k = 1:1
            for j = 1:1 % only one transform per group
                switch klttype
                  case 'kkt'
                    covH = cov(reshape(permute(double(layer_weights),[3,5,1,2,4]),p*g,[])',1);
                    covX = squeeze(X_vars);
                  case 'klt'
                    covH = cov(reshape(permute(double(layer_weights),[3,5,1,2,4]),p*g,[])',1);
                    covX = eye(p*g);
                  case 'idt'
                    covH = eye(p*g);
                    covX = eye(p*g);
                end
                invcovX = inv(covX+covX'+0.01*eye(p*g)*eigs(covX+covX',1));
                [V,d] = eig(covH+covH',invcovX+invcovX','chol','vector');
                invVt = inv(V')./sqrt(sum(inv(V').^2));
                T{l}(:,:,k,1) = inv(invVt(:,end:-1:1));
                T{l}(:,:,k,2) = invVt(:,end:-1:1);
            end
        end
        T{l} = reshape(T{l},[p*g,p*g,1,2]);
        disp(sprintf('%s %s | generated inter transform for layer %03d using %d images',...
                     archname, klttype, l, testsize));
    end        
    save(sprintf('%s_%s_%d_inter',archname,klttype,testsize),'-v7.3','T');
end