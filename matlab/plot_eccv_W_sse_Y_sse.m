clear all; 
close all; 

files = {'../python/resnet18py_idt_sum_50000_output_inter.mat',...
         '../python/resnet18py_klt_sum_50000_output_inter.mat',...
         '../python/resnet18py_klt_sum_50000_output_exter.mat'};

map = get(gca,'colororder');
map = map([4,1,3],:);
map(3,:) = 0.5*[1,1,1];
for f = 1:length(files)
    load(files{f});

    figure(1);
    semilogy(hist_sum_coded,hist_sum_W_sse/sqrt(10),'Color',map(f,:),'LineWidth',0.75);
    axis([1,5,10^-6,10^-4]);
    xticks(1:5);
    yticks([1e-6,3e-6,1e-5,3e-5,1e-4])
    yticklabels({'$10^{-6}$','','$10^{-5}$','','$10^{-4}$'})
    hold on;
    
    figure(2);
    semilogy(hist_sum_coded,hist_sum_Y_sse,'Color',map(f,:),'LineWidth',0.75);
    axis([1,5,10^-1,10^+1]);
    yticks([1e-1,3e-1,1e-0,3e-0,1e+1])
    xticks(1:5);
    hold on;
    
    % figure(3);
    % semilogy(hist_sum_coded,pred_sum_Y_sse,'Color',map(f,:),'LineWidth',0.75);
    % axis([0,4,10^-1,10^+1]);
    % xticks(0:4);
    yticklabels({'$10^{-1}$','','$10^{+0}$','','$10^{+1}$'})
    % hold on;
    
    figure(3);
    plot(hist_sum_coded, 100*hist_sum_Y_top,'Color',map(f,:),'LineWidth',0.75);
    axis([1,5,50,70]);
    xticks(1:5);
    yticks(50:5:70);
    %yticklabels({'50.0','55.0','60.0','65.0','70.0'})
    hold on;
end


for k = 1:3
    figure(k);
    %xticklabels({});
    set(gca,'XMinorGrid','on');
    set(gcf,'Color','none');
    grid on;
    set(gca,'XMinorGrid','off');
    set(gca,'YMinorGrid','off');
    pdfprint(sprintf('temp_%d.pdf',k),'Width',9.5,'Height',9,'Position',[2,1.5,7,7]);
end    
