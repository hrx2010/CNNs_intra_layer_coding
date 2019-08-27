clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
tranname = 'dft2';
testsize = 1024;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);

l_kernel = findconv(neural.Layers);
l_length = length(l_kernel);

klt2 = @(x) gklt(x,adf);
trannames = {'dst2','fst2'};
t_length = length(trannames) + 1;
hist_gainX = cell(l_length,t_length);
hist_gainT = cell(l_length,t_length);
hist_coded = ones(l_length,t_length);

layers = neural.Layers(l_kernel);

for l = 1:l_length
        X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
        [H,W,P,Q] = size(X);
        [h,w,p,q,g] = size(layers(l).Weights);
        T = reshape(permute(layers(l).Weights,[1,2,3,5,4]),[h,w,p*g,q]);
        T_ssd = mean(mean(T.^2,1),2);

        for i = 1:length(trannames)
            tranname = trannames{i};
            trans = str2func(tranname);
            X_psd = diag3(permute(blkfft2(permute(blkfft2(blktoeplitz(autocorr2(X,h,w)),h,w,trans),[2,1,3]),h,w,trans),[2,1,3]));
            T_psd = mean(abs(trans(T)).^2,4)*(1/h/w);

            X_psd(find(X_psd<0)) = NaN;
            hist_meanX{l,i} = mean(X_psd(:));
            hist_geomX{l,i} = geomean(X_psd(:));

            hist_meanT{l,i} = mean(T_psd(:));
            hist_geomT{l,i} = geomean(T_psd(:));
            hist_coded(l,i) = h*w*p*q*g;

            gain_X = hist_meanX{l,i}/hist_geomX{l,i};
            gain_T = hist_meanT{l,i}/hist_geomT{l,i};
            disp(sprintf('Coding gain for layer %03d is %5.2f (%5.2f, %5.2f) dB (%5d coefficients)', ...
                         l, 10*log10(gain_X*gain_T), 10*log10(gain_X), 10*log10(gain_T), hist_coded(l,i)));
        end

        % hist_gainT{l,end} = mean(T_ssd(:))/geomean(T_ssd(:));


        % mean_gainX = sum(cell2mat(hist_gainX{:,i}).*hist_coded{:,i})/sum(hist_coded{:,i});
        % mean_gainT = sum(cell2mat(hist_gainT{:,i}).*hist_coded{:,i})/sum(hist_coded{:,i});
        % mean_gains = sum(cell2mat(hist_gainX{:,i}).*cell2mat(hist_gainT{:,i}).*hist_coded{:,i})/sum(hist_coded{:,i});

        % disp(sprintf('Coding gain for layer all is %5.2f (%5.2f, %5.2f) dB (%5d coefficients)', ...
        %              10*log10(mean_gains), 10*log10(mean_gainX), 10*log10(mean_gainT), sum(hist_coded{:,i})));
end