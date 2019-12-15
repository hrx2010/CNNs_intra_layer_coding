function x = transform_inter(x,T)
    p = size(T,1);
    g = size(T,2)/size(T,1);
    T = reshape(T,p,p,g);
    [h,w,~,q,~] = size(x);
    x = permute(reshape(permute(x,[1,2,3,5,4]),h,w,p,g,[]),[1,2,3,5,4]);
    for k = 1:g
        x(:,:,:,:,k) = permute(reshape(T(:,:,k)*reshape(permute(x(:,:,:,:,k),[3,1,2,4]),[p,h*w*q]),[p,h,w,q]),[2,3,1,4]);
    end
end