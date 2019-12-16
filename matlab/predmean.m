function [X_mean, X_vars] = predmean(neural,images,outlayer,layer_weights)
    X_part = 1000;
    X_size = length(images.Files);
    X_sets = ceil(X_size/X_part);
    X_mean = 0;
    g = size(layer_weights,5);
    for i = 1:X_sets
        X = activations(neural,images.partition(X_sets,i),outlayer,'MiniBatchSize',500);
        [h,w,p,~] = size(X);
        if p ~= size(layer_weights,3)*g
            p = p*h*w; h = 1; w = 1;
        end
        X = reshape(X,h,w,p/g,g,[]) - 0.0000;
        X_mean = X_mean + sum(mean(mean(reshape(X,h,w,p/g,g,[]),1),2),5);
    end
    X_mean = double(X_mean) * (1/X_size);

    X_vars = 0;
    for i = 1:X_sets
        X = activations(neural,images.partition(X_sets,i),outlayer,'MiniBatchSize',500);
        [h,w,p,~] = size(X);
        if p ~= size(layer_weights,3)*g
            p = p*h*w; h = 1; w = 1;
        end
        X = reshape(X,h,w,p/g,g,[]) - X_mean;
        X_vars = X_vars + sum(mean(mean(reshape(X,h,w,p/g,1,g,[]).*reshape(X,h,w,1,p/g,g,[]),1),2),6);
    end
    X_vars = double(X_vars) * (1/X_size);
end
