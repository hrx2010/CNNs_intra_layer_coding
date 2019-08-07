clear all;
close all;

archname = 'alexnet';
testsize = 1024;
trans = {'idt2','dft2'};
for i = 1:length(trans)
    tranname = trans{i};
    load(sprintf('%s_%s_val_zero_%d',archname,tranname,testsize));
    zero_W_sse = hist_W_sse;
    zero_Y_sse = hist_Y_sse;
    zero_Y_top = hist_Y_top;
    zero_coded = hist_coded;
    zero_delta = hist_delta;
    load(sprintf('%s_%s_val_%d_old',archname,tranname,testsize));
    for i = 1:5
        hist_W_sse{i} = [zero_W_sse{i}(1,:,:,:);hist_W_sse{i}];
        hist_Y_sse{i} = [zero_Y_sse{i}(1,:,:,:);hist_Y_sse{i}];
        hist_Y_top{i} = [zero_Y_top{i}(1,:,:,:);hist_Y_top{i}];
        hist_coded{i} = [zero_coded{i}(1,:,:,:);hist_coded{i}];
        hist_delta{i} = [zero_delta{i}(1,:,:,:);hist_delta{i}];
    end
    save(sprintf('%s_%s_val_%d',archname,tranname,testsize),'hist_coded','hist_Y_sse','hist_Y_top','hist_delta','hist_W_sse');
end

