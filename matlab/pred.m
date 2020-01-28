function Y_hat = pred(neural,images,outlayer,batch)
    if nargin < 4
        batch = 25;
    end
    if strcmp('output',outlayer)
        outlayer = neural.Layers(end - 2).Name;
    end
    Y_hat = activations(neural,images,outlayer,'MiniBatchSize',batch);
end