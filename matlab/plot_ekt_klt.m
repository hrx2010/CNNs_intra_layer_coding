clear all;
close all; 

%cmap = viridis(256);
map = get(gca,'colororder');

load ../python/resnet18py_klt_sum_50000_output_inter.mat
%semilogy(hist_sum_coded,hist_sum_Y_sse,'LineWidth',1,'Color',cmap(end*3/4,:))
plot(hist_sum_coded,100*hist_sum_Y_top,'LineWidth',1,'Color',map(1,:));
hold on;

% load alexnetpy_ekt_sum_50000_1_8_output_joint_total_new_new.mat
% plot([5;hist_sum_coded],(100*[.565;hist_sum_Y_top]),'r')


% load alexnetpy_klt_sum_50000_1_8_output_multi_total.mat
% plot(hist_sum_coded,100*hist_sum_Y_top,'k')

load ../python/resnet18py_idt_sum_50000_output_inter.mat 
%semilogy(hist_sum_coded,hist_sum_Y_sse,'LineWidth',1,'Color',cmap(end,:))
plot(hist_sum_coded,100*hist_sum_Y_top,'LineWidth',1,'Color',map(4,:));

%axis([0,4,10^-1,10^+1]);
axis([0,4,50,70]);
xticks(0:1:4);
xticklabels({})
yticks(50:5:70);
%yticklabels({'$10^{-1}$','$10^{+0}$','$10^{+1}$','$10^{+2}$'})
%axis([0.4,0.7,38,54]);
% xticks(0.4:0.1:0.7);
% yticks(38:4:54);

labels = yticklabels;
labels{end} = '$\%$'
yticklabels(labels);

set(gca,'LineWidth',0.5/0.8);
set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',0),'Width',9.75,'Height',9,'Position',[1.75,1.25,7.25,7]);
