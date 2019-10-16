function Y = blktrans(X,T,h,w)
    Y = zeros(size(X));
    for i = 1:size(X,2)
        Y(:,i) = reshape(transform(reshape(X(:,i),h,w,1),T),h*w,1);
    end
end