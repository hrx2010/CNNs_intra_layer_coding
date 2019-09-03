function b=dct(a,varargin)
%DCT  Discrete cosine transform.
%   Y = DCT(X) returns the discrete cosine transform of vector X.
%   If X is a matrix, the DCT operation is applied to each
%   column.  For N-D arrays, DCT operates on the first non-singleton 
%   dimension.  This transform can be inverted using IDCT.
%
%   Y = DCT(X,N) pads or truncates the vector X to length N 
%   before transforming.
%
%   Y = DCT(X,[],DIM) or Y = DCT(X,N,DIM) applies the DCT operation along
%   dimension DIM.
%
%   Y = DCT(...,'Type',K) specifies the type of discrete cosine transform
%   to compute.  K can be one of 1, 2, 3, or 4, to represent the DCT-I,
%   DCT-II, DCT-III, and DCT-IV transforms, respectively.  The default
%   value for K is 2 (the DCT-II transform).
%
%   % Example:
%   %   Find how many DCT coefficients represent 99% of the energy 
%   %   in a sequence.
%
%   x = (1:100) + 50*cos((1:100)*2*pi/40);  % Input Signal 
%   X = dct(x);                             % Discrete cosine transform
%   [XX,ind] = sort(abs(X)); ind = fliplr(ind);
%   num_coeff = 1;
%   while (norm([X(ind(1:num_coeff)) zeros(1,100-num_coeff)])/norm(X)<.99)
%       num_coeff = num_coeff + 1;
%   end;
%   num_coeff                  
%
%   See also FFT, IFFT, IDCT.

%   Author(s): C. Thompson, 2-12-93
%              S. Eddins, 10-26-94, revised
%   Copyright 1988-2018 The MathWorks, Inc.

%   References: 
%   1) A. K. Jain, "Fundamentals of Digital Image
%      Processing", pp. 150-153.
%   2) Wallace, "The JPEG Still Picture Compression Standard",
%      Communications of the ACM, April 1991.


if nargin == 0
  error(message('signal:dct:Nargchk'));
end

if nargin > 1
    [varargin{:}] = convertStringsToChars(varargin{:});
end

% checks if X is a valid numeric data input
signal.internal.sigcheckfloattype(a,'','dct','X');

if isempty(a)
  b = zeros(0,0,'like',a);
  return
end

if (~coder.target('MATLAB') || isa(a,'gpuArray')) && (isempty(varargin) || isscalar(varargin{1}))
  b = dctcodegen(a,varargin{:});
else
  b = dctinternal(a,varargin{:});
end

function b = dctinternal(varargin)

p = inputParser;
p.addRequired('X');
p.addOptional('N',[]);
p.addOptional('DIM',[]);
p.addParameter('Type',2);
p.parse(varargin{:});

r = p.Results;
x = r.X;
type = r.Type;
if isempty(r.N) && isempty(r.DIM)
  [dim,n] = firstNonSingletonDimension(x);
elseif isempty(r.N)
  dim = r.DIM;
  n = size(x,dim);
elseif isempty(r.DIM)
  dim = firstNonSingletonDimension(x);
  n = r.N;
else
  dim = r.DIM;
  n = r.N;
end

validateattributes(n,{'numeric'},{'integer','scalar','positive','finite'});
validateattributes(dim,{'numeric'},{'integer','scalar','positive','finite'});
n = double(n);

scale = sqrt([1/(2*(n-1)) 1/(2*n) 1/(2*n) 1/(2*n)]);
dcscale = sqrt([2 1/2 2 1]);

if type==1
  x = x .* scale(type);
  idc = dimselect(dim,size(x));
  x(1+idc) = x(1+idc) * dcscale(type);
  if size(x,dim) >= n
    x(end-idc) = x(end-idc) * dcscale(type);
  end
elseif type==3
  x = x .* scale(type);
  idc = 1+dimselect(dim,size(x));
  x(idc) = x(idc) * dcscale(type);
end

if n==1 
  if dim==1
    b = matlab.internal.math.transform.mldct(x(1,:),n,dim,'Variant',type);
  elseif dim==2
    b = matlab.internal.math.transform.mldct(x(:,1),n,dim,'Variant',type);
  else
    b = matlab.internal.math.transform.mldct(x,n,dim,'Variant',type);
  end      
else
  b = matlab.internal.math.transform.mldct(x,n,dim,'Variant',type);
end

if type==1
  idc = dimselect(dim,size(b));
  b(idc+1) = b(idc+1) * sqrt(0.5);
  b(end-idc) = b(end-idc) * sqrt(0.5);
elseif type==2
  b = b .* scale(type);
  idc = 1+dimselect(dim,size(b));
  b(idc) = b(idc) * dcscale(type);
elseif type==4
  b = b .* scale(type);
  idc = 1+dimselect(dim,size(b));
  b(idc) = b(idc) * dcscale(type);
end

function idx = dimselect(idim, dim)
ndim = numel(dim);
nel = prod(dim);
dcterm = prod(dim(1:min(idim-1,ndim)));
if idim<=ndim
  nskip = dcterm*dim(idim);
else
  nskip = dcterm;
end
idx = (0:dcterm-1)' + (0:nskip:nel-1);
idx = idx(:);
    
function [dim,n] = firstNonSingletonDimension(a)
sz = size(a);
dim = find(sz~=1,1,'first');
if isempty(dim)
  dim = 1;
  n = 1;
else
  n = sz(dim);
end

function b = dctcodegen(a,n)

% If input is a vector, make it a column:
do_trans = (size(a,1) == 1);
if do_trans
  a = a(:); 
end

if nargin==1
  n = size(a,1);
end
% Cast to enforce precision rules
n = signal.internal.sigcasttofloat(n,'double','dct','N','allownumeric');
m = size(a,2);

% Pad or truncate input if necessary
if size(a,1)<n
  aa = zeros(n,m,'like',a);
  aa(1:size(a,1),:) = a;
else
  aa = a(1:n,:);
end

% Compute weights to multiply DFT coefficients (cast enforces the precision
% rules)
ww = (exp(-1i*cast(0:n-1,'like',a)*pi/(2*n))/sqrt(2*n)).';

ww(1) = ww(1) / sqrt(2);

if rem(n,2)==1 || ~isreal(a) % odd case
  % Form intermediate even-symmetric matrix
  y = zeros(2*n,m,'like',a);
  y(1:n,:) = aa;
  y(n+1:2*n,:) = flipud(aa);
  
  % Compute the FFT and keep the appropriate portion:
  yy = fft(y);  
  yy = yy(1:n,:);

else % even case
  % Re-order the elements of the columns of x
  y = [ aa(1:2:n,:); aa(n:-2:2,:) ];
  yy = fft(y);  
  ww = 2*ww;  % Double the weights for even-length case  
end

% Multiply FFT by weights:
b = ww(:,ones(1,m)) .* yy;

if isreal(a)
  b = real(b);
end
if do_trans
  b = b.';
end
