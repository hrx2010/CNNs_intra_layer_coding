clear all; 
close all; 

files = {'../python/resnet18py_idt_sum_50000_output_inter.mat',...
         '../python/resnet18py_klt_sum_50000_output_inter.mat'};

map = vega20();
set(gcf,'defaultAxesColorOrder',map([8,2],:));
map(3,:) = 0.5*[1,1,1];

for f = 1:length(files)
    load(files{f});

    figure(1);
    plot(hist_sum_coded, 100*hist_sum_Y_top,'LineWidth',0.75);
    hold on;
    axis([1,5,55,70]);
    xticks(1:5);
    yticks(55:5:70);
    %yticklabels({'50.0','55.0','60.0','65.0','70.0'})
    hold on;
end
%plot([1,5],[69.71,69.71],'Color',map(3,:),'LineWidth',0.75);


for k = 1:1
    figure(k);
    %xticklabels({});
    set(gcf,'Color','none');
    %grid on;
    pdfprint(sprintf('temp_%d.pdf',k),'Width',8,'Height',6.75,'Position',[1.5,1.25,6,5]);
end    
