clear all;
close all;

archname = 'vgg16';
tranname = {'klt','kkt'};
modename = {'inter','inter'};
numcoeff = [512,4096,1000];
colorind = [9,10,11,12,13,14,15,16,8,6,5,7,2,4,1,3];
testsize = 50000;
map = vega10;
map(10,:) = 0.5;

for l = 14:16
    for t = 2%1:length(tranname)
        load(sprintf('%s_%s_sum_%d_%d_%d_output_%s_total.mat',archname,tranname{t},testsize,1,16,modename{t}));
        plot(hist_sum_coded,hist_sum_non0s(:,l)/numcoeff(l-13)*100,'Color',map(colorind(l),:));
        hold on;
    end
end

hold off;
axis([0,8,0,100]);

xticks(0:2:8);
yticks(0:25:100);

x = xticklabels;
x{end} = '$R$';
xticklabels(x);

%xticklabels({});

y = yticklabels;
y{end} = '$\%$';
yticklabels(y);

set(gcf,'Color','none');
grid on;
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',l),'Width',10.75,'Height',9,'Position',[1.5,1.5,8.75,7]);
