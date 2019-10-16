function covH = covariances(H,dim)
    [h,w,d] = size(H);
    switch dim
      case 1
        covH = cov(reshape(permute(H(:,:,:),[1,2,3]),[h,w*d])')+...
               cov(reshape(permute(H(:,:,:),[2,1,3]),[w,h*d])');
        covH = kron(covH,covH);
      case 2
        covH = cov(reshape(permute(H(:,:,:),[1,2,3]),[h*w,d])');
    end
    covH = covH' + covH;
end