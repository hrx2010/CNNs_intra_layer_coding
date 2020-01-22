clear all;
close all;

archname = 'alexnet';
tranname = {'idt2','dct2','klt1','kkt1','kkt','klt'};
modename = {'intra','intra','intra','intra','inter','inter'};
colorind = [1,2,3,4,5,7];
testsize = 5000;
map = vega10;

for l = 1
    figure(l);
    for t = 1:length(tranname)
        load(sprintf('%s_%s_sum_%d_%d_%d_output_%s.mat',archname,tranname{t},testsize,1,5,modename{t}));
        plot(hist_sum_coded,100*hist_sum_Y_top,'Color',map(colorind(t),:));
        hold on;
    end
    hold off;
    axis([0,8,40,60]);

    xticks(0:2:8);
    yticks(40:5:60);
    
    x = xticklabels;
    x{end} = '$R$';
    xticklabels(x);
    
    y = yticklabels;
    y{end} = '$\%$';
    yticklabels(y);

    % if l == 1
    %     yticklabels({});
    % end
    % if l == 1
    %     xticklabels({});
    % end
    set(gcf,'Color','none');
    grid on;
    set(gca,'YMinorGrid','off');
    pdfprint(sprintf('temp_%d.pdf',l),'Width',21,'Height',9,'Position',[1.5,1.5,18.75,7]);
end