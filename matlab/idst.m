function a = idst(b,varargin)
%IDST Inverse discrete sine transform.
%   X = IDST(Y) inverts the DST transform, returning the original vector
%   if Y was obtained using Y = DST(X).  If Y is a matrix, the IDST 
%   operation is applied to each column.  For N-D arrays, IDST operates 
%   on the first non-singleton dimension.  
%
%   X = IDST(Y,N) pads or truncates the vector Y to length N 
%   before transforming.
%
%   X = IDST(Y,[],DIM) or X = IDST(Y,N,DIM) applies the IDST operation
%   along dimension DIM.
%
%   X = IDST(...,'Type',K) specifies the type of inverse discrete sine
%   transform to compute.  K can be one of 1, 2, 3, or 4, to represent the
%   IDST-I, IDST-II, IDST-III, and IDST-IV transforms, respectively.  The
%   default value for K is 2 (the IDST-II transform).
%
%   % Example:
%   %   Generate a noisy 25 Hz sinusoidal sequence sampled at 1000 Hz and
%   %   compute the DST of this sequence and reconstruct the signal using 
%   %   only those components with magnitude greater than 0.9
%   
%   t = (0:999)/1000;           % Time vector
%   x = sin(2*pi*25*t);         % Sinusoid
%   x = x + 0.1*randn(1,1000);  % Add noise
%   y = dst(x);                 % Compute DST
%   y(abs(y) < 0.9) = 0;        % remove small components
%   z = idst(y);                % Reconstruct signal w/inverse DST
%   subplot(2,1,1) 
%   plot(t,x)
%   title('Original Signal')
%   subplot(2,1,2)
%   plot(t,z)
%   title('Reconstructed Signal')
%
%   See also FFT, IFFT, DCT.

%   Author(s): C. Thompson, 2-12-93
%              S. Eddins, 10-26-94, revised
%   Copyright 1988-2018 The MathWorks, Inc.

%   References: 
%   1) A. K. Jain, "Fundamentals of Digital Image
%      Processing", pp. 150-153.
%   2) Wallace, "The JPEG Still Picture Compression Standard",
%      Communications of the ACM, April 1991.

if nargin == 0
  error(message('signal:idst:Nargchk'));
end

if nargin > 1
    [varargin{:}] = convertStringsToChars(varargin{:});
end

% checks if X is a valid numeric data input
signal.internal.sigcheckfloattype(b,'','idst','Y');

if isempty(b)
  a = zeros(0,0,'like',b);
  return
end

if (~coder.target('MATLAB') || isa(b,'gpuArray')) && (isempty(varargin) || isscalar(varargin{1}))
  a = idstcodegen(b,varargin{:});
else
  a = idstinternal(b,varargin{:});
end

function a = idstinternal(varargin)

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

scale = sqrt([2*(n+1) 2*n 2*n 2*n]);
dcscale = sqrt([1/2 2 1/2 1]);

if type==1
  x = x .* scale(type);
elseif type==2
  x = x .* scale(type);
  idc = 1+dimselect(dim,size(x));
  x(idc) = x(idc) * dcscale(type);
elseif type==4
  x = x .* scale(type);
end
% if type==1
%   idc = dimselect(dim,size(x));
%   x(idc+1) = x(idc+1) * sqrt(2);
%   if size(x,dim)>=n
%     x(end-idc) = x(end-idc) * sqrt(2);
%   end
% elseif type==2
%   x = x .* scale(type);
%   idc = 1+dimselect(dim,size(x));
%   x(idc) = x(idc) * dcscale(type);
% elseif type==4
%   x = x .* scale(type);
%   idc = 1+dimselect(dim,size(x));
%   x(idc) = x(idc) * dcscale(type);
% end

if n==1 
  if dim==1
    a = matlab.internal.math.transform.mlidst(x(1,:),n,dim,'Variant',type);
  elseif dim==2
    a = matlab.internal.math.transform.mlidst(x(:,1),n,dim,'Variant',type);
  else
    a = matlab.internal.math.transform.mlidst(x,n,dim,'Variant',type);
  end      
else
  a = matlab.internal.math.transform.mlidst(x,n,dim,'Variant',type);
end

% if type==1
%   a = a .* scale(type);
%   idc = dimselect(dim,size(a));
%   a(1+idc) = a(1+idc) * dcscale(type);
%   a(end-idc) = a(end-idc) * dcscale(type);
if type==3
  a = a .* scale(type);
  idc = 1+dimselect(dim,size(a));
  a(idc) = a(idc) * dcscale(type);
end

function idx = dimselect(idim, dim)
ndim = numel(dim);
nel = prod(dim);
dsterm = prod(dim(1:min(idim-1,ndim)));
if idim<=ndim
  nskip = dsterm*dim(idim);
else
  nskip = dsterm;
end
idx = (0:dsterm-1)' + (0:nskip:nel-1);
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


function a = idstcodegen(b,n)

% If input is a vector, make it a column:
do_trans = (size(b,1) == 1);
if do_trans
    b = b(:); 
end
   
if nargin==1
  n = size(b,1);
end
m = size(b,2);

% Cast to enforce precision rules. 
n = signal.internal.sigcasttofloat(n,'double','idst','N','allownumeric');

% Pad or truncate b if necessary
if size(b,1)<n
  bb = zeros(n,m,'like',b);
  bb(1:size(b,1),:) = b;
else
  bb = b(1:n,:);
end

% Compute wieghts
ww = sqrt(2*n) * exp(1i*cast(0:n-1,'like',b)*pi/(2*n)).';

if rem(n,2)==1 || ~isreal(b) % odd case
  % Form intermediate even-symmetric matrix.
  ww(1) = ww(1) * sqrt(2);
  W = ww(:,ones(1,m));
  yy = zeros(2*n,m,'like',b);
  yy(1:n,:) = W.*bb;
  yy(n+2:2*n,:) = -1i*W(2:n,:).*flipud(bb(2:n,:));
  
  y = ifft(yy);

  % Extract inverse DST
  a = y(1:n,:);

else % even case
  % Compute precorrection factor
  ww(1) = ww(1)/sqrt(2);
  W = ww(:,ones(1,m));
  yy = W.*bb;

  % Compute x tilde using equation (5.93) in Jain
  y = ifft(yy);
  
  % Re-order elements of each column according to equations (5.93) and
  % (5.94) in Jain
  a = zeros(n,m,'like',y);
  a(1:2:n,:) = y(1:n/2,:);
  a(2:2:n,:) = y(n:-1:n/2+1,:);
end

if isreal(b)
  a = real(a); 
end
if do_trans
  a = a.';
end
