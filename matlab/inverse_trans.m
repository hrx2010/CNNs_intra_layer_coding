function x = inverse_trans(x,K)
    x = reshape(K*reshape(x,size(K,1),[]),size(x));
end