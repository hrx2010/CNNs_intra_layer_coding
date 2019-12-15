function S = xvars2(X,T,h,w)
    [H,W,P,Q] = size(X);
    X = flip(flip(X,1),2);
    S = zeros(h,w,P);% as many shifts as there are pixels
    for i = 0:H-1
        for j = 0:W-1
            r = mod((0:h-1) + i, H) + 1;
            c = mod((0:w-1) + j, W) + 1;
            S = S + sum(abs(transform_intra(X(r,c,:,:),T)).^2,4);
        end
    end
    S = S * (1/H/W/Q);
end