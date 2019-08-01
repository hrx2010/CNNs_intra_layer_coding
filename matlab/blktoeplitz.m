function T = blktoeplitz(c)
%TOEPLITZ Toeplitz matrix.
%   BLKTOEPLITZ(C) is a non-symmetric Toeplitz matrix having C as its
%   first column and R as its first row.
%
%   Class support for inputs C,R:
%      float: double, single
%      integer: uint8, int8, uint16, int16, uint32, int32, uint64, int64
%
%   See also HANKEL.

%   Thanks to A.K. Booer for the original version.
%   Copyright 1984-2017 The MathWorks, Inc.

[p,q,r] = size(c);
x = [c(p:-1:2,:,:);c];                  % build vector of user data
i = (0:p-1)' + (p:-1:1);                % Toeplitz subscripts
t = reshape(x(i,:,:),p,p,[]);           % actual data

T = reshape(mat2cell(reshape(t,p,[]),p,p*ones(1,q*r)),[1,q,r]);
X = [T(:,q:-1:2,:),T];                  % build vector of user data
j = (0:q-1)' + (q:-1:1);                % Toeplitz subscripts
T = cell2mat(reshape(X(:,j,:),q,q,[])); % actual data
