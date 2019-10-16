function R = autocorr1(X,M)
    X = X - mean(X(:));
    R = zeros(M,1,1,size(X,4));
    for m = 1:M
        R(m,1) = mean(reshape(circshift(X,[m,1] - 1).*X,[],1));
    end
end