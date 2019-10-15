function T = blktoeplitz(c)
%BLKTOEPLITZ Bock Toeplitz matrix.
%   BLKTOEPLITZ(C) Creates an NM-by-NM block Toeplitz matrix from an
%   N-by-M matrix C, whose p,qth block element is an N-by-N Toeplitz
%   matrix created from the (abs(q-p)+1)th column of C.
%
%   Class support for inputs C,R:
%      float: double, single
%      integer: uint8, int8, uint16, int16, uint32, int32, uint64, int64
%

[p,q] = size(c);
x = [c(p:-1:2,:);c];                  % build vector of user data
i = (0:p-1)' + (p:-1:1);              % Toeplitz subscripts
t = reshape(x(i,:),p,p,q);           % actual data

T = mat2cell(reshape(t,p,p*q),p,p*ones(1,q));
X = [T(:,q:-1:2),T];                  % build vector of user data
j = (0:q-1)' + (q:-1:1);                % Toeplitz subscripts
T = cell2mat(reshape(X(:,j),q,q)); % actual data
