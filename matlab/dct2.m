function x = dct2(x)
    x = dct(dct(x,[],1,'Type',1),[],2,'Type',1);
end