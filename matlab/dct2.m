function x = dct2(x,type)
    x = dct(dct(x,[],1,'Type',type),[],2,'Type',type);
end