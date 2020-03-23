clear all; 
close all; 

%load alexnet_idt_val_100_output_inter_kern.mat


map = vega10;
map(10,:) = 0.5;
k = 1:9; 
l = 3;
load alexnet_idt_val_100_3_3_output_inter_kern_mean_removed.mat
y_sse_removed = min(squeeze(kern_Y_sse{l}(k,:,1)'),[],1);
y_top_removed = min(squeeze(kern_Y_top{l}(k,:,1)'),[],1);
w_sse_removed = min(squeeze(kern_W_sse{l}(k,:,1)'),[],1);
delta_removed = min(squeeze(kern_delta{l}(k,:,1)'),[],1);

load alexnet_idt_val_100_3_3_output_inter_kern_mean_not_removed.mat
y_sse_not_removed = min(squeeze(kern_Y_sse{l}(k,:,1)'),[],1);
y_top_not_removed = min(squeeze(kern_Y_top{l}(k,:,1)'),[],1);
w_sse_not_removed = min(squeeze(kern_W_sse{l}(k,:,1)'),[],1);
delta_not_removed = min(squeeze(kern_delta{l}(k,:,1)'),[],1);

semilogy(k-1,y_sse_removed/1000/4,'r');
hold on;
semilogy(k-1,y_sse_not_removed/1000/4,'Color',0.5*[1,1,1]);
hold on;
%semilogy(k-1,w_sse_removed/4,'b');

xticks(0:2:6);
yticks(10.^(-5:1:-2));
labels = yticklabels;
%labels{5} = '$10^{+0}$';
%yticklabels(labels);
axis([0,6,10^-5,10^-2]);

set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',k),'Width',9.75,'Height',9,'Position',[2,1.5,7,7]);
    
