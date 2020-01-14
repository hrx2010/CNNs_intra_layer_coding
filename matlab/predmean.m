function [X_mean, X_vars] = predmean(neural,images,outlayer,groups,inputs)
    X_part = 50;
    X_size = length(images.Files);
    X_sets = ceil(X_size/X_part);
    X_mean = 0;
    for i = 1:X_sets
        X = activations(neural,images.partition(X_sets+1,i),outlayer,'MiniBatchSize',X_part);
        [h,w,p,~] = size(X);
        h = h*sqrt(p/(groups*inputs));
        w = w*sqrt(p/(groups*inputs));
        p = inputs;
        g = groups;
        X = reshape(X,h,w,p,g,[]) - 0.0000;
        X_mean = X_mean + sum(mean(mean(reshape(X,h,w,p,g,[]),1),2),5);
    end
    X_mean = double(X_mean) * (1/X_size);

    X_vars = 0;
    for i = 1:X_sets
        X = activations(neural,images.partition(X_sets+1,i),outlayer,'MiniBatchSize',X_part);
        [h,w,p,~] = size(X);
        h = h*sqrt(p/(groups*inputs));
        w = w*sqrt(p/(groups*inputs));
        p = inputs;
        g = groups;
        X = reshape(X,h,w,p,g,[]) - X_mean;
        X_vars = X_vars + sum(mean(mean(reshape(X,h,w,p,1,g,[]).*reshape(X,h,w,1,p,g,[]),1),2),6);
    end
    X_vars = double(X_vars) * (1/X_size);
end
