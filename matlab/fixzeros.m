clear all;
close all;

archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/ILSVRC2012_test_00000*.JPEG';
testsize = 128;
load([archname,'_freq_',num2str(testsize)],'hist_freq_coded','hist_freq_Y_sse','hist_freq_delta');

for j = 1:121
    for k = 1:128
        idx = find(isnan(hist_freq_Y_sse(:,j,k)));
        hist_freq_Y_sse(idx,j,k) = NaN;
        hist_freq_coded(idx,j,k) = NaN;
        hist_freq_delta(idx,j,k) = NaN;
    end
end

save([archname,'_freq_',num2str(testsize)],'hist_freq_coded','hist_freq_Y_sse','hist_freq_delta');