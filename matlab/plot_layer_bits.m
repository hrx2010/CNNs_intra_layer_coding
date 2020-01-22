clear all; 
close all; 

load alexnet_kkt_sum_10_1_8_output_inter_total.mat
map = vega10;
map(10,:) = 0.5;
ind = [8,6,5,7,2,4,1,3,9,10,11,12,13,14,15,16];
%ind = [9,10,11,12,13,14,15,16,8,6,5,7,2,4,1,3];
A = area(hist_sum_coded, hist_sum_total/8/1e6,'LineStyle','none');
for i = 1:length(A)
    a = A(i);
    a.FaceColor = map(ind(i),:);
    a.FaceAlpha = 0.5;
end
axis([0,8,0,80]);
xticks(0:2:8);
yticks(0:20:80);

x = xticklabels;
x{end} = '$R$';
xticklabels(x);
xticklabels({});
y = yticklabels;
y{end} = '$\mathrm{MB}$';
yticklabels(y);

set(gcf,'Color','none');
grid on;
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',0),'Width',10.75,'Height',9,'Position',[1.5,1.5,8.75,7]);
