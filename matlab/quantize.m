function xhat = quantize(x,delta)
    if delta == 0
        xhat = x;
    else
        xhat = delta*(floor(double(x)/delta+0.5));
    end
end