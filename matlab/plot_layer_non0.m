clear all;
close all;

load('alexnetpy_klt_sum_5_1_8_output_inter_total.mat');
plot(hist_sum_coded,(256-hist_sum_non0s(:,6))/256*100,'Color',0.0*[1,1,1]);
hold on;
plot(hist_sum_coded,(4096-hist_sum_non0s(:,7))/4096*100,'Color',0.5*[1,1,1]);
hold on;
plot(hist_sum_coded,(4096-hist_sum_non0s(:,8))/1000*100,'Color','r');

hold off;
axis([0,4,0,100]);

xticks(0:1:4);
yticks(0:25:100);

x = xticklabels;
y = yticklabels;
y{end} = '$\%$';
yticklabels(y);

set(gcf,'Color','none');
grid on;
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',0),'Width',9.75,'Height',9,'Position',[2,1.25,7,7]);

plot(hist_sum_coded,(64-hist_sum_non0s(:,2))/64*100,'Color',0.0*[1,1,1]);
hold on;
% plot(hist_sum_coded,(192-hist_sum_non0s(:,3))/192*100,'Color',0.0*[1,1,1]);
% hold on;
plot(hist_sum_coded,(384-hist_sum_non0s(:,4))/384*100,'Color',0.5*[1,1,1]);
hold on;
plot(hist_sum_coded,(256-hist_sum_non0s(:,5))/256*100,'Color','r');

axis([0,1.2,0,100]);

yticks(0:25:100);
xticks(0:0.3:1.2);
y = yticklabels;
y{end} = '$\%$';
yticklabels(y);

set(gcf,'Color','none');
grid on;
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',0),'Width',9.75,'Height',9,'Position',[1.5,1.25,7,7]);
