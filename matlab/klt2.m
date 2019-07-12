function k = klt2(x)
    [h,w,p,q] = size(x);
    k = reshape(x,[h*w,p*q]);
    C = cov(k');
    [V,D] = eig(C);
    k = reshape(V'*k,[h,w,p,q]);
end