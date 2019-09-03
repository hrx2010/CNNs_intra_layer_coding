function x = idst2(x)
    types = 2 - mod(size(x),2);
    x = idst(idst(x,[],2,'Type',types(2)),[],1,'Type',types(1));
end