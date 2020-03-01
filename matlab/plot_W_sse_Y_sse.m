clear all; 
close all; 

load ../matlab/alexnet_idt_val_1000_output_inter_kern.mat

map = get(gca,'colororder');
map = map([4,1,3],:);
map(3,:) = 0.5*[1,1,1];

k = 9; 
y_sse = squeeze(kern_Y_sse{1}(k,:,1)');
w_sse = squeeze(kern_W_sse{1}(k,:,1)');
delta = squeeze(kern_delta{1}(k,:,1)');
y_sse = [y_sse(1);y_sse];
w_sse = [w_sse(1);w_sse];
delta = [delta(1)-1;delta];
[min_y_sse,arg_y_sse] = min(y_sse);
[min_w_sse,arg_w_sse] = min(w_sse);

loglog(2.^delta,y_sse/1000/4,'Color',map(1,:),'LineWidth',0.75);
hold on;
loglog(2.^delta,w_sse/4,'Color',0.5*[1,1,1],'LineWidth',0.75);
hold on;
loglog(2.^delta(arg_y_sse),min_y_sse/1000/4,'.','MarkerSize',6,'Color',map(1,:))
hold on;
loglog(2.^delta(arg_w_sse),min_w_sse/4,'.','MarkerSize',6,'Color',0.5*[1,1,1],'LineWidth',0.75);

axis([10^-4,10^-1,10^-7,10^-3]);

set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',k),'Width',9.75,'Height',9,'Position',[2,1.5,7,7]);
    
