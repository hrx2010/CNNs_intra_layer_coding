function points = lambda2points(X,Y,Z,lambda)
%LAMBDA2POINTS Find indices at the the specified trade-off lambda
%   POINTS = LAMBDA2POINTS(X,Y,Z,LAMBDA) finds the index of the 
%   rates X, distortions Y for the given distortion-rate tradeoff
%   LAMBDA.
%
% Copyright 2018-2019 Stanford University, Stanford CA 94305
% 
%                         All Rights Reserved
% 
% Permission to use, copy, modify, and distribute this software and
% its documentation for any purpose other than its incorporation into
% a commercial product is hereby granted without fee, provided that
% the above copyright notice appear in all copies and that both that
% copyright notice and this permission notice appear in supporting
% documentation, and that the name of Stanford University not be used
% in advertising or publicity pertaining to distribution of the
% software without specific, written prior permission.
% 
% STANFORD UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
% SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
% FITNESS FOR ANY PARTICULAR PURPOSE.  IN NO EVENT SHALL STANFORD
% UNIVERSITY BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
% DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA
% OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
% TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
% PERFORMANCE OF THIS SOFTWARE.

    lambda = max(0,lambda);
    points = zeros(size(Z,2),size(Z,3))*NaN;
    % compute the convex hulls
    for i = 1:size(X(:,:),2)
        if all(isnan(X(:,i)))
            continue
        end
        k = rdhull(X(:,i),Y(:,i));
        l = [-diff(Y(k,i))./diff(X(k,i));0];
        %find the first point with slope less than lambda
        z = Z(k,i);
        points(i) = z(find(l<=lambda,1));
    end
end