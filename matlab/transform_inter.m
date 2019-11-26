function x = transform_inter(x,T)
    [h,w,p,q,g] = size(x);
    for k = 1:g
        Tk = T(:,(k-1)*end/g+(1:end/g));
        x(:,:,:,:,k) = permute(reshape(Tk*reshape(permute(x(:,:,:,:,k),[3,1,2,4]),[p,h*w*q]),[p,h,w,q]),[2,3,1,4]);
    end
end