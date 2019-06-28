clear all;
close all;

load weight_values;


l = 1; %layer

xcoeffs = fft2(squeeze(weight_values{l}));
xcoeffs = fftshift(fftshift(xcoeffs,1),2);

h = size(xcoeffs,1);
w = size(xcoeffs,2);

for m = 1:3
    for n = 1:3
        % use imaginary or real parts of the coefficients
        if (h*(n-1)+m < (h*w+1)/2)
            xcoeffs(m,n,:,:) = imag(xcoeffs(m,n,:,:));
        else
            xcoeffs(m,n,:,:) = real(xcoeffs(m,n,:,:));
        end
    end
end

psd = mean(mean(xcoeffs.^2,3),4);
bar3c(psd,1);
xlabel('$n$');
ylabel('$m$');
zlabel('$\omega^2(n,m)$');
axis([0.5,3.5,0.5,3.5,0,max(psd(:))]);
axis square;



% F x G different curves, see if RD curves 
% Input layer first.


