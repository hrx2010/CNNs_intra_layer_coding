function x = fct2(x)
    x = dct(dct(circshift(x,[size(x,1)+1,size(x,2)+1]/2),[],1,'Type',1),[],2,'Type',1);
end