function D = diag3(X)
    D = zeros(size(X,1),1,size(X,3));
    for i = 1:size(X,3)
        D(:,1,i) = diag(X(:,:,i));
    end
end