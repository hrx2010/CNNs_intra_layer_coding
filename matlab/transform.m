function x = transform(x,K)
    switch class(K)
      case 'function_handle'
        x = K(x);
      otherwise
        x = reshape(K*reshape(x,size(K,1),[]),size(x));
    end
end