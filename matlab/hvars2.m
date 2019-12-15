function S = hvars2(H,T,h,w)
    [~,~,p,q,g] = size(H);
    S = mean(reshape(permute(transform_intra(H,T),[1,2,3,5,4]).^2,h,w,p*g,q),4);
end