clear all; 
close all; 

map = vega20();
set(gcf,'defaultAxesColorOrder',map([2,8,16],:));
%colors = [2,8,7];

figure(1);
load ../python/resnet50py_klt_sum_5_output_inter.mat
weights = prod(dims_sum_input(:,3:4)-dims_sum_weight(:,3:4)+1,2)';

denom = min(prod(dims_sum_weight(:,[1,3,4]),2),prod(dims_sum_weight(:,2),2));
gains = sum(weights.*hist_sum_non0s,2)./sum(weights.*hist_sum_non0s(1,:),2);
plot(hist_sum_coded,100*gains,'Color',map(7,:),'LineWidth',0.75);
axis([1,4,60,100]);
xticks(1:4);
yticks(60:10:100);

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
    pdfprint(sprintf('temp_%d.pdf',k),'Width',9,'Height',7.5,'Position',[2,1.5,6.5,5.5]);
end    
