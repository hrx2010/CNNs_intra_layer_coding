function [Y_hat,Y_cat] = pred(neural,nclass,images,outlayer)
    if nargin == 3
        outlayer = neural.Layers(end).Name;
    end
    inputdim = nclass.Layers(1).InputSize;
    numparts = images.numpartitions(gcp);
    Y_hat = cell(numparts,1);
    Y_cat = cell(numparts,1);
    parfor p = 1:numparts
        Y_hat{p} = activations(neural,images.partition(numparts,p),outlayer,'OutputAs','column','MiniBatchSize',256);
        Y_cat{p} = classify(nclass,Y_hat{p}(1:inputdim,:))';
    end
    Y_hat = horzcat(Y_hat{:});
    Y_cat = vertcat(Y_cat{:});
end