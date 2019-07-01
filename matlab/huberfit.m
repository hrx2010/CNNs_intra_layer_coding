function Y_hat = huberfit(X,Y,slope)
    m = size(X,1);
    n = size(X,2);

    if nargin == 2
        % get the slope (should be around 6.01)
        x = X(~isnan(X));
        y = 10*log10(Y(~isnan(Y)));
        A = [x(:),ones(length(x(:)),1)];
        p = A\y;
        slope = p(1);
    end

    Y_hat = zeros(m,n);
    for j = 1:n
        sigma = mean(10*log10(Y(:,j)) - slope*X(:,j),'omitnan');
        Y_hat(:,j) = slope*X(:,j) + sigma;
    end
end