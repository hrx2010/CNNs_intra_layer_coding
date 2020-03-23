clear all; 
close all; 

files = {'../python/resnet50py_klt_sum_50000_output_inter.mat'};

map = vega20();
%set(gcf,'defaultAxesColorOrder',map([7,1],:));
colors = [7,1];

for f = 1:length(files)
    load(files{f});

    figure(1);
    plot([0,5],[76.1,76.1],'Color',map(15,:),'LineWidth',0.5);
    hold on;
    plot(hist_sum_coded, 100*hist_sum_Y_top,'Color',map(1,:),'LineWidth',0.75);
    axis([1,5,58,78]);
    xticks(1:1:5);
    yticks(58:5:78);
    %yticklabels({'50.0','55.0','60.0','65.0','70.0'})
    hold on;
end

plot([1.1,1.5,1.9,3.1,5],[72.7,74.1,75.2,76,76],'Color',map(7,:),'LineWidth',0.75);



k = 1;
figure(k);
%xticklabels({});
set(gca,'XMinorGrid','on');
set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
%pdfprint(sprintf('temp_%d.pdf',k),'Width',10,'Height',8,'Position',[2,1.5,7.5,6]);
pdfprint(sprintf('temp_%d.pdf',k),'Width',10,'Height',7.5,'Position',[2,1.5,7.5,5.5]);

