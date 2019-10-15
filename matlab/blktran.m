function T = blktran(T,m,n,tran)
    for i = 1:size(T,2)
        T(:,i,:) = reshape(transform(reshape(T(:,i,:),m,n,[]),tran),m*n,1,[]);
    end
end