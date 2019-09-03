function x = transform(x,T)
    switch class(T)
      case 'function_handle'
        x = T(x);
      otherwise
        x = reshape(T*reshape(x,size(T,1),[]),size(x));
    end
end