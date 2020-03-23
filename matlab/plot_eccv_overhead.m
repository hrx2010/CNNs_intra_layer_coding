clear all;
close all;

overheads = [0.10    43.54
             11.11   11.11
             11.11   11.11
             11.11   11.11
             11.11   11.11
             5.56    22.22
             11.11   11.11
             50.00   50.00
             11.11   11.11
             11.11   11.11
             5.56    22.22
             11.11   11.11
             50.00   50.00
             11.11   11.11
             11.11   11.11
             5.56    22.22
             11.11   11.11
             50.00   50.00
             11.11   11.11
             11.11   11.11
             51.20   51.20];

map = vega20();
set(gcf,'defaultAxesColorOrder',map([2,8],:));
bar((1:21)-0.225,overheads(:,1),0.45,'LineStyle','none');
hold on;
bar((1:21)+0.225,overheads(:,2),0.45,'LineStyle','none');
xticks(1:5:21);
yticks(0:25:75);
yticklabels(0:25:75);
xticklabels(0:5:20);
axis([0.5,21.5,0,75]);


%xticklabels({});
set(gcf,'Color','none');
grid on;
pdfprint(sprintf('temp_%d.pdf',1),'Width',12,'Height',7.5,'Position',[2,1.5,9.5,5.5]);

