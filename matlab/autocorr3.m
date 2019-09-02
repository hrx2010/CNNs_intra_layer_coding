function R = autocorr3(X,M,N)
    X = X - mean(X(:));
    R = zeros(M,N);
    for k = 1:M*N
        [m,n] = ind2sub([M,N],k);
        R(m,n) = mean(reshape(circshift(X,[m,n] - 1).*X,[],1));
    end
end