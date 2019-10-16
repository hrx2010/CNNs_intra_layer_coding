function S = hvars2(H,T,h,w)
    S = sum(transform(H(:,:,:),T).^2,3);
end