function x = forward_trans(x,K)
    x = reshape(inv(K)*reshape(x,size(K,1),[]),size(x));
end