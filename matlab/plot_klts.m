clear all;
close all; 

load alexnetpy_klt_sum_50000_1_8_output_joint_total.mat
plot(hist_sum_coded,100*hist_sum_Y_top,'Color',0.5*[1,1,1])
hold on;

load alexnetpy_klt_sum_50000_1_8_output_multi_total.mat
plot(hist_sum_coded,100*hist_sum_Y_top,'r')


% load alexnetpy_ekt_sum_50000_1_8_output_joint_total_new_new.mat
% plot(hist_sum_coded,100*hist_sum_Y_top,'k')

%axis([0,4,54,57]);

axis([0.4,1.6,39,57]);
xticks(0.4:0.1:0.8);
yticks(39:2:57);

labels = yticklabels;
labels{end} = '$\%$'
yticklabels(labels);

set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',0),'Width',9.75,'Height',9,'Position',[1.75,1.25,7.25,7]);
