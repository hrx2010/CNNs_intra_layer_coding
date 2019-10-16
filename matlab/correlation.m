function covX = correlation(X,dim,s)
    [h,w,d] = size(X);
    switch dim
      case 1
        covX = autocorr1(reshape(permute(X(:,:,:),[1,2,3]),h,w*d),s)+...
               autocorr1(reshape(permute(X(:,:,:),[2,1,3]),w,w*d),s);
        covX = toeplitz1(covX);
        covX = kron(covX,covX);
      case 2
        covX = autocorr3(reshape(permute(X(:,:,:),[1,2,3]),h,w,d),s,s);
        covX = toeplitz2(covX);
    end
    covX = covX' + covX;
end