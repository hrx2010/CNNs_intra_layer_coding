function Y_hat = pred(neural,images,outlayer)
    if strcmp('output',outlayer)
        outlayer = neural.Layers(end - 2).Name;
    end
    Y_hat = activations(neural,images,outlayer,'MiniBatchSize',500);
end