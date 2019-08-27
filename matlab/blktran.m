function T = blkfft2(T,m,n,tran)
    for i = 1:size(T,2)
        T(:,i,:) = reshape(tran(reshape(T(:,i,:),m,n,[])),m*n,1,[]);
    end
end