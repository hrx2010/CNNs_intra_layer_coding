function x = ifft2split(x)
    s = size(x);
    b = prod(s(1:2));
    x = reshape(x,b,[]);
    x(1:(b+1)/2-1,:) = sqrt(1/2)*x(1:(b+1)/2-1,:);
    x((b+1)/2+1:b,:) = sqrt(1/2)*x((b+1)/2+1:b,:);
    x(1:(b+1)/2-1,:) = 1*real(x(1:(b+1)/2-1,:)) - i*real(x(b:-1:(b+1)/2+1,:));
    x((b+1)/2+1:b,:) = i*real(x((b+1)/2+1:b,:)) + 1*real(x((b+1)/2-1:-1:1,:));
    x = reshape(x,s);
end