clear all;
close all;

archname = 'alexnet';
testsize = 8;

load([archname,'_sum_',num2str(testsize)]);
load([archname,'_base_',num2str(testsize)]);

vega = vega10(10);
figure(1);
loglog(10.^[-6,0],10.^[-6,0],'Color',0.0*[1,1,1]);
hold on; 
loglog(mean(pred_sum_Y_sse(1:end,:,:),3)/1000, ...
       mean(hist_sum_Y_sse(1:end,:,:),3)/1000,'.','MarkerSize',12,'Color',vega(6,:));
xticks(10.^(-6:1:0));
yticks(10.^(-6:1:0));
axis([10^-6,10^0,10^-6,10^0]);
xlabel('Predicted MSE');
ylabel('Actual MSE');
%pdfprint('temp1.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);

figure(2);
semilogy(mean(hist_sum_coded(1:end,:,:),3),mean(hist_sum_Y_sse(1:end,:,:),3)/1000,'.-','MarkerSize',8,'Color',vega(6,:));
hold on;
semilogy(mean(hist_base_sum_coded(1:end,:,:),3),mean(hist_base_sum_Y_sse(1:end,:,:),3)/1000,'.-','MarkerSize',8,'Color',vega(7,:));
grid on;
yticks(10.^(-6:2:2));
xticks(0:1:8);
axis([0,8,10^-6,10^2]);
xlabel('Rate (bits)');
ylabel('Actual MSE');
pdfprint('temp2.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);

figure(3);
plot(mean(hist_sum_coded(1:end,:,:),3),mean(hist_sum_Y_sse(1:end,:,:),3)/1000,'.-','MarkerSize',8,'Color',vega(6,:));
hold on;
plot(mean(hist_base_sum_coded(1:end,:,:),3),mean(hist_base_sum_Y_sse(1:end,:,:),3)/1000,'.-','MarkerSize',8,'Color',vega(7,:));
grid on;
xticks(0:1:4);
yticks(0:5:20);
axis([0,4,0,20]);
xlabel('Rate (bits)');
ylabel('Actual MSE');
pdfprint('temp3.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);