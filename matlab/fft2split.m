function x = fft2split(x)
    s = size(x);
    b = prod(s(1:2));
    x = reshape(x,b,[]);
    % x(1:(b+1)/2-1,:) = 2*x(1:(b+1)/2-1,:);
    % x((b+1)/2+1:b,:) = 2*x((b+1)/2+1:b,:);
    x(1:(b+1)/2-1,:) = real(x(1:(b+1)/2-1,:));
    x((b+1)/2+1:b,:) = imag(x((b+1)/2+1:b,:));
    x = reshape(x,s);
end