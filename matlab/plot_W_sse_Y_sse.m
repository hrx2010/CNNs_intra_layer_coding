clear all; 
close all; 

load alexnet_idt_val_100_output_inter_kern.mat

map = vega10;
map(10,:) = 0.5;

k = 3; 
y_sse = squeeze(kern_Y_sse{1}(k,:,1)');
w_sse = squeeze(kern_W_sse{1}(k,:,1)');
delta = squeeze(kern_delta{1}(k,:,1)');
loglog(2.^delta,y_sse/1000/4,'Color',0.5*[1,1,1]);
hold on;
loglog(2.^delta,w_sse/4,'Color',0*[1,1,1]);

axis([10^-3,10^-0,10^-5,10^-1]);

set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',k),'Width',9.75,'Height',9,'Position',[2,1.5,7,7]);
    
