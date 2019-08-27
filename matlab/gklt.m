function [V,D] = gklt(H, covX)
    invcovX = inv(covX);
    [V,D] = eig(cov(H'),(invcovX+invcovX')/2);
end