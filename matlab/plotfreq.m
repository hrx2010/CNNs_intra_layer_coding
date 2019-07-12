clear all;
close all;

archname = 'alexnet';
testsize = 128;
load([archname,'_freq_',num2str(testsize)]);

figure(1);
loglog(mean(hist_freq_delta(:,1:61,:),3),mean(hist_freq_Y_sse(:,1:61,:),3)/1000);
xticks(10.^(-3:1));
yticks(10.^(-10:2:0));
xlabel('Quantization step-size');
ylabel('Mean squared-error');
axis([10^-3,10^1,10^-10,10^0]);
grid on;
pdfprint('temp1.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);

figure(2);
semilogy(mean(hist_freq_coded(:,1:61,:),3),mean(hist_freq_Y_sse(:,1:61,:),3)/1000);
% xticks(10.^(-3:1));
% yticks(10.^(-10:2:0));
xlabel('Rate (bits)');
ylabel('Mean squared-error');
axis([0,8,10^-12,10^0]);
grid on;
pdfprint('temp1.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);
