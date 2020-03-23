clear all; 
close all; 

load alexnetpy_ekt_sum_50000_1_8_output_joint_total.mat
plot([5;hist_sum_coded],(100*[.565;hist_sum_Y_top]),'Color',0.5*[1,1,1]);
hold on;

load alexnetpy_klt_sum_50000_1_8_output_inter_total.mat
plot([5;hist_sum_coded],(100*[.565;hist_sum_Y_top]),'Color',0.0*[1,1,1]);
hold on;

load alexnetpy_klt_sum_50000_1_8_output_multi_total.mat
plot([5;hist_sum_coded],(100*[.565;hist_sum_Y_top]),'Color','r');
hold on;

plot([0,5],[56.6,56.6],'Color',0.5*[1,1,1]);
axis([0,4,54,57]);
xticks(0:1:4);
yticks(54:1:57);

% axis([0.4,0.7,38,54]);
% xticks(0.4:0.1:0.7);
% yticks(38:4:54);


set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',0),'Width',9.75,'Height',9,'Position',[1.75,1.25,7.25,7]);
