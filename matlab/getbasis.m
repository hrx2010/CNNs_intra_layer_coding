function [K,D] = getbasis(H,X,tranname)
    [h,w,p,q] = size(H);
    switch tranname
      case 'gklt'
        covX = blktoeplitz(autocorr3(X,h,w));
        invcovX = inv(covX);
        covH = cov(reshape(H,[h*w,p*q])');
        [K,D] = eig(covH,(invcovX+invcovX')/2);
    end
end