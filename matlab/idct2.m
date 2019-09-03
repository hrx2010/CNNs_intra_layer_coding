function x = idct2(x,type)
    x = idct(idct(x,[],2,'Type',type),[],1,'Type',type);
end