clear all; 
close all; 

files = {'../python/resnet18py_klt_sum_50000_output_inter.mat'};

map = vega20();
%set(gcf,'defaultAxesColorOrder',map([7,1],:));
colors = [7,1];

% %INQ
% plot([5,4,3,2],(100-[31.02,31.11,31.92,33.98]),'.','MarkerSize',6,'Color',map(16,:),'LineWidth',0.75);
% hold on;
% plot([5,4,3,2],(100-[31.02,31.11,31.92,33.98]),'-','MarkerSize',6,'Color',map(16,:),'LineWidth',0.75);
% %BWN TWN TTQ
% plot([1,2,2],[60.80,61.80,66.6],'.','MarkerSize',6,'Color',map(16,:),'LineWidth',0.75);
% %LQN
% plot([2,3,4],[68,69.3,70],'.','MarkerSize',6,'Color',map(16,:),'LineWidth',0.75);
% plot([2,3,4],[68,69.3,70],'-','MarkerSize',6,'Color',map(16,:),'LineWidth',0.75);

for f = 1:length(files)
    load(files{f});

    figure(1);
    plot([0,5],[69.7,69.7],'Color',map(15,:),'LineWidth',0.5);
    hold on;
    plot(hist_sum_coded, 100*hist_sum_Y_top,'Color',map(1,:),'LineWidth',0.75);
    axis([1,5,58,78]);
    xticks(1:1:5);
    yticks(58:5:78);
    %labels = yticklabels;
    %labels{end} = '$\%$'
    %yticklabels(labels)
    %yticklabels({'50.0','55.0','60.0','65.0','70.0'})
    hold on;
end

%OURS
plot([1,1.4,1.9,3.0,3.5,5.0],[64.1,66.7,68.1,69.5,69.6,69.7],'Color',map(7,:),'LineWidth',0.75);


k = 1 ;
figure(k);
%xticklabels({});
set(gca,'XMinorGrid','on');
set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
%pdfprint(sprintf('temp_%d.pdf',k),'Width',10,'Height',8,'Position',[2,1.5,7.5,6]);
pdfprint(sprintf('temp_%d.pdf',k),'Width',10,'Height',7.5,'Position',[2,1.5,7.5,5.5]);
