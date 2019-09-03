function T = gettrans(H,X,tranname)
    T = cell(2,1);
    [h,w,p,q] = size(H);
    switch tranname
      case 'klt2'
        covX = blktoeplitz(autocorr3(X,h,w));
        invcovX = inv(covX);
        covH = cov(reshape(H,[h*w,p*q])');
        K = eig(covH,(invcovX+invcovX')/2);
        T{1} = inv(K);
        T{2} = K;
      case 'dct2'
        T{1} = @(x) dct2(x,2-mod(h,2));
        T{2} = @(x) idct2(x,2-mod(h,2));
      case 'dst2'
        T{1} = @(x) dst2(x,2-mod(h,2));
        T{2} = @(x) idst2(x,2-mod(h,2));
      case 'dft2'
        T{1} = @dft2;
        T{2} = @idft2;
      case 'idt2'
        T{1} = @idt2;
        T{2} = @iidt2;
    end
end
