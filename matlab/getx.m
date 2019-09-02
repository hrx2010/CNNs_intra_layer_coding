function X = getx(neural,nclass,images,outlayer)
    outlayer = findinput(neural,outlayer);
    inputdim = nclass.Layers(1).InputSize;
    numparts = images.numpartitions(gcp);
    Y_hat = cell(1,1,1,numparts);
    Y_cat = cell(1,1,1,numparts);
    parfor p = 1:numparts
        Y_hat{p} = activations(neural,images.partition(numparts,p),outlayer,'OutputAs','channel','MiniBatchSize',32);
    end
    X = cell2mat(Y_hat);
end