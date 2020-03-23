clear all;
close all;

map = vega20();
set(gcf,'defaultAxesColorOrder',map([7,1],:));
map(3,:) = 0.5*[1,1,1];

l = 14;
rate = 4;
load(sprintf(['../python/resnet18py_klt_val_%03d_0100_output_inter_kern_full.mat'],l));

plot(kern_delta(rate+1,:,1),log10(kern_Y_sse(rate+1,:,1))-2,'LineWidth',0.75,'Color',map(7,:));
hold on;
[kern_Y_min,i] = min(kern_Y_sse(rate+1,:,1));
plot(kern_delta(rate+1,i,1),log10(kern_Y_min)-2,'.','MarkerSize',6,'Color',map(7,:));

plot(kern_delta(rate+1,:,1),log10(kern_W_sse(rate+1,:,1))+1,'LineWidth',0.75,'Color',map(1,:));
hold on;
[kern_W_min,i] = min(kern_W_sse(rate+1,:,1));
plot(kern_delta(rate+1,i,1),log10(kern_Y_sse(rate+1,i,1))-2,'.','MarkerSize',6,'Color',map(8,:));

plot([kern_delta(rate+1,i,1),kern_delta(rate+1,i,1)],[-8,0],'Color',map(8,:));
plot(kern_delta(rate+1,i,1),log10(kern_W_min)+1,'.','MarkerSize',6,'Color',map(1,:));

xticks(-16:4:0);
yticks(-4:1:-1);
%axis([-5,-2,])
axis([-16,0,-4,-1]);

grid on;
set(gcf,'Color','none');
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',l),'Width',10,'Height',7.5,'Position',[2,1.5,7.25,5.5]);

