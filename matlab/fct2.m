function c = fct2(x)
    c = dct(dct(x,[],1),[],2);
end