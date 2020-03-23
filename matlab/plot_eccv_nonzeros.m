clear all; 
close all; 

files = {'../python/resnet18py_klt_sum_5_output_inter.mat',...
         '../python/resnet34py_klt_sum_5_output_inter.mat'};

map = vega20();
set(gcf,'defaultAxesColorOrder',map([2,8,16],:));
%colors = [2,8,7];

figure(1);
load ../python/resnet34py_klt_sum_5_output_inter.mat
rates = [26,21];
bar(0:36,[hist_sum_non0s(rates(1),:);diff(hist_sum_non0s(rates,:))]'./hist_sum_non0s(1,:)'*100,0.9,'stacked','LineStyle','none');
xticks(0:5:36);
yticks(0:25:100);
axis([-0.5,36.5,0,100])
% layer = 10
% for f = 1:length(files)
%     load(files{f});

%     figure(1);
%     plot(hist_sum_coded,hist_sum_non0s/hist_sum_non0s,'Color',map(colors(f),:),'LineWidth',0.75);
%     axis([1,5,-1,+1]);
%     xticks(1:1:5);
%     yticks(-1:0.5:1);
%     hold on;
    
%     % figure(3);
%     % semilogy(hist_sum_coded,pred_sum_Y_sse,'Color',map(f,:),'LineWidth',0.75);
%     % axis([0,4,10^-1,10^+1]);
%     % xticks(0:4);
%     %yticklabels({'$10^{-1}$','','$10^{+0}$','','$10^{+1}$'})
%     % hold on;
    
%     figure(2);
%     plot([0,5],[69.27,69.27],'Color',map(16,:),'LineWidth',0.5);
%     hold on;
%     plot(hist_sum_coded, 100*hist_sum_Y_top,'Color',map(colors(f),:),'LineWidth',0.75);
%     axis([1,5,50,70]);
%     xticks(1:1:5);
%     yticks(50:5:75);
%     %yticklabels({'50.0','55.0','60.0','65.0','70.0'})
%     hold on;
% end


for k = 1:1
    figure(k);
    %xticklabels({});
    set(gca,'XMinorGrid','on');
    set(gcf,'Color','none');
    grid on;
    set(gca,'XMinorGrid','off');
    set(gca,'YMinorGrid','off');
    %pdfprint(sprintf('temp_%d.pdf',k),'Width',10,'Height',8,'Position',[2,1.5,7.5,6]);
    pdfprint(sprintf('temp_%d.pdf',k),'Width',17.5,'Height',7.5,'Position',[2,1.5,15,5.5]);
end    
