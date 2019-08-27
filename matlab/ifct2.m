function x = ifct2(x)
    x = idct(idct(x,[],2,'Type',1),[],1,'Type',1);
end