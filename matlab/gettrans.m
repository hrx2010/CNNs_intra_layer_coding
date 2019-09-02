function T = gettrans(H,X,tranname)
    T = cell(2,1);
    switch tranname
      case 'gklt'
        [h,w,p,q] = size(H);
        covX = blktoeplitz(autocorr3(X,h,w));
        invcovX = inv(covX);
        covH = cov(reshape(H,[h*w,p*q])');
        [K,D] = eig(covH,(invcovX+invcovX')/2);
        T{1} = inv(K);
        T{2} = K;
      case 'dct2'
        T{1} = @dct2;
        T{2} = @idct2;
      case 'dft2'
        T{1} = @dft2;
        T{2} = @idft2;
      case 'idt2'
        T{1} = @idt2;
        T{2} = @iidt2;
    end
end