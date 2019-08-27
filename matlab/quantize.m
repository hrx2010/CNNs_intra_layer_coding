function xhat = quantize(x,delta,B)
    if B ~= 0
        minpoint = -2^(B-1)*delta;
        maxpoint = +2^(B-1)*delta - delta;
    else
        minpoint = 0;
        maxpoint = 0;
    end

    xhat = min(maxpoint,max(minpoint,delta*floor(x/delta + 0.5)));
end