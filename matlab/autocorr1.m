function R = autocorr1(X,M)
% X = X - mean(X(:));
    R = zeros(M,1,size(X,3),size(X,4));
    for m = 1:M
        R(m,1,:,:) = 0.5 * (mean(mean(circshift(X,[m-1,0]).*X,1),2) + ...
                            mean(mean(circshift(X,[0,m-1]).*X,1),2));
    end
end