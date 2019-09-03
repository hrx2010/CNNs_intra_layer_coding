function x = idst2(x,type)
    x = idst(idst(x,[],2,'Type',type),[],1,'Type',type);
end