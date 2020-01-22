clear all;
close all;

archname = 'alexnet';
tranname = {'idt2','dct2','klt1','kkt1','kkt','klt'};
modename = {'intra','intra','intra','intra','inter','inter'};
colorind = [10,2,3,5,4,1];
% tranname = {'idt','klt','kkt'};
% modename = {'inter','inter','inter'};
% colorind = [10,1,4];
testsize = 5000;
map = vega10;
map(10,:) = 0.5;

for l = 4
    for t = 1:length(tranname)
        load(sprintf('%s_%s_sum_%d_%d_%d_output_%s.mat',archname,tranname{t},testsize,l,l,modename{t}));
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
    
    y = yticklabels;
    y{end} = '$D(R)$';
    yticklabels(y);

    % if l ~= 1 && l ~= 5
        yticklabels({});
    % end
    % if l ~= 5
        xticklabels({});
    % end
    set(gcf,'Color','none');
    grid on;
    set(gca,'YMinorGrid','off');
    pdfprint(sprintf('temp_%d.pdf',l),'Width',12,'Height',9,'Position',[2.25,1.5,9.25,7]);
end


