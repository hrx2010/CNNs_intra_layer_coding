clear all; 
close all; 

load alexnet_idt_val_1000_output_inter_kern.mat

map = vega10;
%map(10,:) = 0.5;

k = 2:1:9; 
y_sse = squeeze(kern_Y_sse{1}(k,:,1)');
w_sse = squeeze(kern_W_sse{1}(k,:,1)');
delta = squeeze(kern_delta{1}(k,:,1)');
% loglog(2.^delta,y_sse/1000/4,'Color',0.5*[1,1,1]);
% hold on;
% [~,i] = min(y_sse/1000/4,[],1);
% i = sub2ind(size(y_sse),i,1:8);
% loglog(2.^delta(i),y_sse(i)/1000/4,'Color','r');
% hold on;
% loglog(2.^delta(i),y_sse(i)/1000/4,'.','Color','r');
%

loglog(2.^delta,w_sse/4,'Color',0.25*[1,1,1]);
hold on;
[~,i] = min(w_sse/4,[],1);
i = sub2ind(size(w_sse),i,1:8);
loglog(2.^delta(i),w_sse(i)/4,'Color','b');
hold on; 
loglog(2.^delta(i),w_sse(i)/4,'.','Color','b');


axis([10^-4,10^-1,10^-7,10^-3]);

set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',k),'Width',9.75,'Height',9,'Position',[2,1.5,7,7]);

