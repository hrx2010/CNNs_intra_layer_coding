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
colorind = [10,1,4];
testsize = 50000;
map = vega10;
map(10,:) = 0.5;

for l = 0
    %figure(l);
    for t = [1,3]%1:length(tranname)
        load(sprintf('%s_%s_sum_%d_%d_%d_output_%s_total.mat',archname,tranname{t},testsize,6,8,modename{t}));
        semilogy(hist_sum_coded,hist_sum_Y_sse,'Color',map(colorind(t),:));
        hold on;
    end
    hold off;
    axis([0,8,10^-4,10^+2]);

    xticks(0:2:8);
    yticks(10.^(-4:2:+2));
    yticklabels({'$10^{-4}$','$10^{-2}$','$10^{+0}$','$10^{+2}$'});
    
    x = xticklabels;
    x{end} = '$R$';
    xticklabels(x);
    xticklabels({});
    y = yticklabels;
    y{end} = '$D(R)$';
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
    pdfprint(sprintf('temp_%d.pdf',l),'Width',12,'Height',9,'Position',[2.25,1.5,9.25,7]);
end