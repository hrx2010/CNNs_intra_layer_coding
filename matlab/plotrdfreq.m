clear all;
close all;

archname = 'alexnet';
testsize = 1024;
trans = {'idt2','dft2'};%,'dct2'};
X = cell(length(trans),1);
Y = cell(length(trans),1);
Z = cell(length(trans),1);

for i = 1:length(trans)
    tranname = trans{i};
    load(sprintf('%s_%s_sum_layer_1_%d',archname,tranname,testsize));
    X{i} = mean(hist_sum_coded,3);
    Y{i} = mean(hist_sum_Y_sse,3);
    Z{i} = mean(hist_sum_Y_top,3);
end
X = horzcat(X{:});
Y = horzcat(Y{:});
Z = horzcat(Z{:});

figure(1);
semilogy(X,Y,'.-','MarkerSize',8);
xticks(0:1:8);
yticks(10.^(-4:2:2));
axis([0,6,10^-4,10^2]);
xlabel('Rate (bits)');
ylabel('Mean squared error');
pdfprint('temp2.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);

figure(2);
plot(X,100*Z,'.-','MarkerSize',8);
hold on;
plot(X,100*Z(1,1)*ones(size(X,1),1),'-','Color',0.5*[1,1,1]);
xticks(0:1:8);
yticks(0:10:60);
axis([0,6,0,60]);
xlabel('Rate (bits)');
ylabel('Top-1 accuracy (\%)');
pdfprint('temp3.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);
