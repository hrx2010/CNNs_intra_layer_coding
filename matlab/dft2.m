function x = dft2(x)
    [h,w,d] = size(x);
    x = sqrt(1/h/w)*fft2split(fftshift(fftshift(fft2(x),1),2));
end