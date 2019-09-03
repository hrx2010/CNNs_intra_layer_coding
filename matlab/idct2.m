function x = idct2(x)
    types = 2 - mod(size(x),2);
    x = idct(idct(x,[],2,'Type',types(2)),[],1,'Type',types(1));
end