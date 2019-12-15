function x = dct2(x, types)
    if nargin < 2
        types = 2 - mod(size(x),2);
    end
    x = dct(dct(x,[],1,'Type',types(1)),[],2,'Type',types(2));
end