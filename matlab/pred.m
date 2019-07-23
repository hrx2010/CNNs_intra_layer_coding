function [Y_hat,Y_cat] = pred(neural,nclass,images)
    numparts = images.numpartitions(gcp);
    Y_hat = cell(numparts,1);
    Y_cat = cell(numparts,1);
    parfor p = 1:numparts
        Y_hat{p} = predict(neural,images.partition(numparts,p))';
        Y_cat{p} = classify(nclass,Y_hat{p})';
    end
    Y_hat = horzcat(Y_hat{:});
    Y_cat = vertcat(Y_cat{:});
end