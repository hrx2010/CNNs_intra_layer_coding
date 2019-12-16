function [X_mean, X_vars] = predmean(neural,images,outlayer,layer_weights)
    X_part = 250;
    X_size = length(images.Files);
    X_sets = ceil(X_size/X_part);
    X_mean = 0;
    g = size(layer_weights,5);
    for i = 1:X_sets
        X = activations(neural,images.partition(X_sets,i),outlayer,'MiniBatchSize',250);
        [h,w,p,~] = size(X);
        X = reshape(X,h,w,p/g,g,[]) - 0.0000;
        X_mean = X_mean + sum(mean(mean(reshape(X,h,w,p/g,g,[]),1),2),5);
    end
    X_mean = double(X_mean) * (1/X_size);

    X_vars = 0;
    for i = 1:X_sets
        X = activations(neural,images.partition(X_sets,i),outlayer,'MiniBatchSize',250);
        [h,w,p,~] = size(X);
        X = reshape(X,h,w,p/g,g,[]) - X_mean;
        X_vars = X_vars + sum(mean(mean(reshape(X,h,w,p/g,1,g,[]).*reshape(X,h,w,1,p/g,g,[]),1),2),6);
    end
    X_vars = double(X_vars) * (1/X_size);
end
