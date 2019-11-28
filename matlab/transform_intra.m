function x = transform_intra(x,T)
    switch class(T)
      case 'function_handle'
        x = T(x);
      otherwise
        [h,w,p,q,g] = size(x);
        for j = 1:g
            for i = 1:p
                x(:,:,i,:,j) = reshape(T{i,j}*reshape(x(:,:,i,:,j),h*w,q),h,w,1,q,1);
            end
        end
    end
end