clear all;
close all;

archname = 'alexnet';
testsize = 8;

load([archname,'_sum_',num2str(testsize)]);

figure(1);
loglog(mean(pred_sum_Y_sse(1:end,:),3)/1000,mean(hist_sum_Y_sse(1:end,:),3)/1000,'.');
hold on;
loglog(10.^[-7,1],10.^[-7,1]);
xticks(10.^(-7:2:1));
yticks(10.^(-7:2:1));
axis([10^-7,10^1,10^-7,10^1],'square');
xlabel('Predicted MSE');
ylabel('Actual MSE');

% figure(2);
% semilogy(mean(hist_sum_coded(1:end,:,:),3),mean(hist_sum_Y_sse(1:end,:,:),3)/1000,'.');
