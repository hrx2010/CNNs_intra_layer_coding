function x = dst2(x,type)
    x = dst(dst(x,[],1,'Type',type),[],2,'Type',type);
end