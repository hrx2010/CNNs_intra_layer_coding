clear all;
close all; 

load alexnetpy_ekt_sum_50000_1_8_output_joint_total_new_new.mat
plot([5;hist_sum_coded],(100*[.565;hist_sum_Y_top]/.566),'r')
hold on;
load alexnetpy_idt_sum_50000_1_8_output_inter_total.mat 
plot(hist_sum_coded,100*hist_sum_Y_top./0.566,'Color',0.5*[1,1,1])
hold on;
plot([0,4],[100,100],'Color',0.5*[1,1,1]);

plot(2,57.5/57.2*100,'.','Color',0*[1,1,1],'MarkerSize',8) %TTQ
plot(2,54.5/57.2*100,'.','Color',0*[1,1,1],'MarkerSize',8) %TWN
plot(1,53.9/57.2*100,'.','Color',0*[1,1,1],'MarkerSize',8) %DOREFA
plot(1,56.8/56.6*100,'.','Color',0*[1,1,1],'MarkerSize',8) %BWN
plot(2,60.5/61.8*100,'.','Color',0*[1,1,1],'MarkerSize',8) %LQ-NETS

axis([0,4,92,102]);
xticks(0:1:4);
yticks(92:4:102);




% axis([0.4,0.7,38,54]);
% xticks(0.4:0.1:0.7);
% yticks(38:4:54);

% labels = yticklabels;
% labels{end} = '$\%$'
% yticklabels(labels);

set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',0),'Width',13.5,'Height',9,'Position',[1.5,1.25,7.5+3.75,7]);

