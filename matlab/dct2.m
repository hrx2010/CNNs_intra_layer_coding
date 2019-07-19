function x = dct2(x)
    x = dct(dct(x,[],1),[],2);
end