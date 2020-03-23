clear all;
close all;

for rate = 1:10
    for l = 1:20
        load(sprintf(['../python/resnet18py_klt_val_%03d_0100_output_inter_kern.mat'],l));
        [~,i] = min(squeeze(kern_W_sse(rate+1,:,1)));
        [~,j] = min(squeeze(kern_Y_sse(rate+1,:,1)));
        disp(sprintf(['optimal step-size for layer %03d at %2d bits: ' ...
                      '%2d, %2d (%2d)'],l,rate,i,j,i-j));
    end
end


