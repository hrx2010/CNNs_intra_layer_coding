function S = xvars2(X,T,h,w)
    X = flip(flip(X,1),2);
    H = size(X,1);
    W = size(X,2);
    S = zeros(h,w);
    for i = 0:H-1
        for j = 0:W-1
            r = mod((0:h-1) + i, H) + 1;
            c = mod((0:w-1) + j, W) + 1;
            S = S + sum(abs(transform(X(r,c,:),T)).^2,3);
        end
    end
end