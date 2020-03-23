clear all;
close all;

load ../python/resnet34py_inter_klt_gain_10000.mat
layers = length(areag);
l = 37;

len = length(areaw{l});
map = vega20();
set(gcf,'defaultAxesColorOrder',map([2,8],:));

area(0:len-1,[areaw{l}',areag{l}'-0.5],'LineStyle','none');

xticks(0:len/4:len);
yticks(-16:8:32);
%xticklabels(0:128:512);
axis([0,len,-16,24]);


%xticklabels({});
set(gcf,'Color','none');
grid on;
pdfprint(sprintf('temp_%d.pdf',1),'Width',10,'Height',7.5,'Position',[2,1.5,7.25,5.5]);

