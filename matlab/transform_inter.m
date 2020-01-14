function x = transform_inter(x,T)
    p = size(T,1);
    g = size(T,2)/size(T,1);
    T = reshape(T,p,p,g);
    [h,w,P,q,~] = size(x);
    if p ~= P %p is the number of hannels of the kernels
              %according to the transform, and it should
              %be 3 or 48 or something like that
        h = sqrt(P/p); % 
        w = sqrt(P/p);
        x = reshape(x,[h,w,p,q,g]);
    end
    x = permute(reshape(permute(x,[1,2,3,5,4]),h,w,p,g,[]),[1,2,3,5,4]);
    for k = 1:g
        x(:,:,:,:,k) = permute(reshape(T(:,:,k)*reshape(permute(x(:,:,:,:,k),[3,1,2,4]),[p,h*w*q]),[p,h,w,q]),[2,3,1,4]);
    end
    if p ~= P
        x = reshape(x,[1,1,P,q,g]);
    end
end