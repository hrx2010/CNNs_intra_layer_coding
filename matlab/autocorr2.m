function R = autocorr2(X,M,N)
    X = X - mean(mean(mean(X,1),2),4);
    P = size(X,3);
    R = zeros(M,N,P);
    for k = 1:round(M*N/2)
        [m,n] = ind2sub([M,N],k);
        R(m,n,:) = mean(mean(mean(circshift(X,[m,n]-round([M,N]/2)).*X,1),2),4);
        R(M+1-m,N+1-n,:) = R(m,n,:);
    end
    R = ifftshift(ifftshift(R,1),2);
end