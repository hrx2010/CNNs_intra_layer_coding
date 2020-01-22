clear all;
close all;

% archname = 'alexnet';
% tranname = {'idt2','dct2','klt1','kkt1','kkt','klt'};
% modename = {'intra','intra','intra','intra','inter','inter'};
% colorind = [1,2,3,4,5,7];
% testsize = 5000;
% map = vega10;

archname = 'alexnet';
% tranname = {'idt2','dct2','klt1','kkt1','kkt','klt'};
% modename = {'intra','intra','intra','intra','inter','inter'};
% colorind = [10,2,3,5,4,1];
tranname = {'idt','klt','kkt'};
modename = {'inter','inter','inter'};
colorind = [10,1,4,2,3,5,6,7,8,9];
testsize = 50000;
map = vega10;
map(10,:) = 0.5;

for l = 1
    %figure(l);
    for t = [1,3]%1:length(tranname)
        load(sprintf('%s_%s_sum_%d_%d_%d_output_%s_total.mat',archname,tranname{t},testsize,6,8,modename{t}));
        plot(hist_sum_coded,100*hist_sum_Y_top,'Color',map(colorind(t),:));
        hold on;
    end

    plot([0,8],[57.,57.1],'Color',map(10,:));
    total = 1e6/60954656*8;
    sizes = [12.7,12.6, 7.6, 8.9,-4.8,15.2]; %Less is more, Circulant CNN, Quantized CNN
    top1s = [-0.4,-1.4, 0.1, 0.0, 0.7, 0.4]
    for i = 1:6
        plot(sizes(i)*total,57.1+top1s(i),'.','Color',map(colorind(i+3),:));
    end

    hold off;
    axis([0,4,40,60]);

    xticks(0:1:4);
    yticks(40:5:60);
    
    x = xticklabels;
    x{end} = '$R$';
    xticklabels(x);
    xticklabels({});
    y = yticklabels;
    y{end} = '$\%$';
    yticklabels(y);

    % if l == 1
        % yticklabels({});
    % end
    % if l ~= 1
    %     xticklabels({});
    % end
    set(gcf,'Color','none');
    grid on;
    set(gca,'YMinorGrid','off');
    pdfprint(sprintf('temp_%d.pdf',0),'Width',10.75,'Height',9,'Position',[1.5,1.5,8.75,7]);
end

