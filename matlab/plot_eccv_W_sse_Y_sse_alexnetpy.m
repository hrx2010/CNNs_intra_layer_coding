clear all; 
close all; 

files = {'../python/alexnetpy_idt_sum_50000_output_inter.mat',...
         '../python/alexnetpy_klt_sum_50000_output_exter.mat',...
         '../python/alexnetpy_klt_sum_50000_output_inter.mat'};

map = vega20();
set(gcf,'defaultAxesColorOrder',map([7,1],:));
colors = [1,8,7];

for f = 1:length(files)
    load(files{f});

    % figure(1);
    % plot(hist_sum_coded,log10(hist_sum_W_sse/sqrt(10)),'Color',map(f,:),'LineWidth',0.75);
    % axis([1,5,10^-6,10^-4]);
    % xticks(0:0.5:2);
    % yticks([1e-6,3e-6,1e-5,3e-5,1e-4])
    % yticklabels({'$10^{-6}$','','$10^{-5}$','','$10^{-4}$'})
    % hold on;
    
    figure(1);
    plot(hist_sum_coded,log10(hist_sum_Y_sse),'Color',map(colors(f),:),'LineWidth',0.75);
    axis([0,3,-2,1]);
    xticks(0:1:3);
    yticks(-2:1);
    hold on;
    
    % figure(3);
    % semilogy(hist_sum_coded,pred_sum_Y_sse,'Color',map(f,:),'LineWidth',0.75);
    % axis([0,4,10^-1,10^+1]);
    % xticks(0:4);
    %yticklabels({'$10^{-1}$','','$10^{+0}$','','$10^{+1}$'})
    % hold on;
    
    figure(2);
    plot([0,8],[55.71,55.71],'Color',map(16,:),'LineWidth',0.75);
    hold on;
    plot(hist_sum_coded, 100*hist_sum_Y_top,'Color',map(colors(f),:),'LineWidth',0.75);
    axis([0,3,52,56]);
    xticks(0:1:3);
    yticks(52:1:56);
    %yticklabels({'50.0','55.0','60.0','65.0','70.0'})
    hold on;
end


for k = 1:2
    figure(k);
    %xticklabels({});
    set(gca,'XMinorGrid','on');
    set(gcf,'Color','none');
    grid on;
    set(gca,'XMinorGrid','off');
    set(gca,'YMinorGrid','off');
    pdfprint(sprintf('temp_%d.pdf',k),'Width',10,'Height',8,'Position',[2,1.5,7.5,6]);
end    
