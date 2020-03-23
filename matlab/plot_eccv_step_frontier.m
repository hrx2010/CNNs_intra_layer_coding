clear all;
close all;

% l = 9;
% for rate = 1:10
%     
%     [~,i] = min(squeeze(kern_W_sse(rate+1,:,1)));
%     [~,j] = min(squeeze(kern_Y_sse(rate+1,:,1)));
%     disp(sprintf(['optimal step-size for layer %03d at %2d bits: ' ...
%     '%2d, %2d (%2d)'],l,rate,i,j,i-j));
% end


map = vega20();
set(gcf,'defaultAxesColorOrder',map([7,1],:));
map(3,:) = 0.75*[1,1,1];

l = 14;
load(sprintf(['../python/resnet18py_klt_val_%03d_0100_output_inter_kern_full.mat'],l));
rate = 1:8
plot(kern_delta(rate+1,:,1)',log10(kern_Y_sse(rate+1,:,1))'-2,'LineWidth',0.75,'Color',map(3,:));
hold on;
[kern_Y_min,i] = min(kern_Y_sse(rate+1,:,1),[],2);
plot(kern_delta(1,i,1),log10(kern_Y_min)-2,'LineWidth',0.75,'Color',map(7,:));
plot(kern_delta(1,i,1),log10(kern_Y_min)-2,'.','MarkerSize',6,'Color',map(7,:));

[kern_W_min,i] = min(kern_W_sse(rate+1,:,1),[],2);
kern_Y_min = [];
for j = rate
    kern_Y_min = [kern_Y_min, kern_Y_sse(j+1,i(j),1)];
end
plot(kern_delta(1,i,1),log10(kern_Y_min)-2,'LineWidth',0.75,'Color',map(8,:));
plot(kern_delta(1,i,1),log10(kern_Y_min)-2,'.','MarkerSize',6,'Color',map(8,:));



% plot(kern_delta(rate+1,:,1),log10(kern_W_sse(rate+1,:,1))+1,'LineWidth',0.75,'Color',map(1,:));
% hold on;
% plot(kern_delta(rate+1,i,1),log10(kern_W_min)+1,'.','MarkerSize',6,'Color',map(1,:));

% plot([kern_delta(rate+1,i,1),kern_delta(rate+1,i,1)],[-8,0],'Color',map(7,:));

xticks(-16:4:0);
yticks(-6:1:-1);
axis([-16,0,-6,-1]);

grid on;
set(gcf,'Color','none');
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',l),'Width',10,'Height',7.5,'Position',[2,1.5,7.25,5.5]);

