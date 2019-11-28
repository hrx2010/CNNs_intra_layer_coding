function x = transform_inter(x,T)
    [h,w,p,q,g] = size(x);
    T = reshape(T,[p,p,g]);
    for k = 1:g
        x(:,:,:,:,k) = permute(reshape(T(:,:,k)*reshape(permute(x(:,:,:,:,k),[3,1,2,4]),[p,h*w*q]),[p,h,w,q]),[2,3,1,4]);
    end
end