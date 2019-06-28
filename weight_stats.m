clear all;
close all;

load weight_values;


l = 12; %layer
m = 2;  %row
n = 2;  %col

% IM IM RE
% IM DC RE
% IM RE RE

xcoeffs = fft2(squeeze(weight_values{l}(:,:,:,:)));
xcoeffs = fftshift(fftshift(xcoeffs,1),2);
weights = weight_values{l};

h = size(xcoeffs,1);
w = size(xcoeffs,2);

% use imaginary or real parts of the coefficients
if (h*(n-1)+m < (h*w+1)/2)
    x = reshape(imag(xcoeffs(m,n,:,:)),[],1);
else
    x = reshape(real(xcoeffs(m,n,:,:)),[],1);
end

%x = reshape(real(weights(m,n,:,:)),[],1);
bin_width = 3.5*std(x(:))*numel(x)^(-1/3);

bin_count = 256;
bin_edges = (-bin_count/2+0.5:bin_count/2-0.5)*bin_width;
bin_point = (-bin_count/2+1.0:bin_count/2-1.0)*bin_width;

x(x<bin_edges(1)) = bin_edges(1);
x(x>bin_edges(end)) = bin_edges(end);

histcount = histcounts(x,bin_edges);
plot(bin_point,histcount);
xlabel(sprintf('Layer %d. Coeff. %d,%d.', l, m, n));
ylabel('Frequency');
%histogram(x,bin_edges','Normalization','pdf');
disp(sprintf('kurtosis: %f', kurtosis(x)));
disp(sprintf('skewness: %f', skewness(x)));
