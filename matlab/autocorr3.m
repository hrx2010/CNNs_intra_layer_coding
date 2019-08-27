function R = autocorr3(X,M,N)
    X = X - mean(mean(mean(X,1),2),4);
    P = size(X,3);
    R = zeros(M,N,P);
    for k = 1:M*N
        [m,n] = ind2sub([M,N],k);
        R(m,n,:) = mean(mean(mean(circshift(X,[m,n]-1).*X,1),2),4);
    end
end