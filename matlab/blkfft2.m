function T = blkfft2(T,m,n)
    for i = 1:size(T,2)
        T(:,i,:) = reshape(fft2(reshape(T(:,i,:),m,n,[])),m*n,1,[]);
    end
    T = T*sqrt(1/m/n);
end