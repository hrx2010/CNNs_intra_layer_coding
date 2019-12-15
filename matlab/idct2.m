function x = idct2(x, types)
    if nargin < 2
        types = 2 - mod(size(x),2);
    end
    x = idct(idct(x,[],2,'Type',types(2)),[],1,'Type',types(1));
end