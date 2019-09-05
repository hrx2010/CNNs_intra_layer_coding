clear all;
close all;

archname = 'vgg16';
testsize = 1000;
outlayer = 'output';
inlayers = 1:13;
trans = {'dct2','dst2'};

X = cell(length(trans),1);
Y = cell(length(trans),1);
Z = cell(length(trans),1);

for i = 1:length(trans)
    tranname = trans{i};
    load(sprintf('%s_%s_sum_%d_%d_%d_%s',archname,tranname,testsize,inlayers(1),inlayers(end),outlayer));
    X{i} = mean(hist_sum_coded,3);
    Y{i} = mean(hist_sum_Y_sse,3);
    Z{i} = mean(hist_sum_Y_top,3);
end
X = horzcat(X{:});
Y = horzcat(Y{:});
Z = horzcat(Z{:});

figure(1);
semilogy(X,Y,'.-','MarkerSize',8);
xticks(0:8);
yticks(10.^(-4:1:2));
axis([0,8,10^-4,10^2]);
xlabel('Rate (bits)');
ylabel('Mean squared error');
pdfprint('temp2.pdf','Width',14,'Height',12,'Position',[3,2.5,10,8.5]);

figure(2);
plot(X,100*Z/Z(1,1),'.-','MarkerSize',8);
% hold on;
% plot(X,100*Z(1,1)*ones(size(X,1),1),'-','Color',0.5*[1,1,1]);
xticks(0:8);
yticks(0:20:100);
axis([0,8,0,100]);
xlabel('Rate (bits)');
ylabel('Relative accuracy (\%)');
pdfprint('temp3.pdf','Width',14,'Height',12,'Position',[3,2.5,10,8.5]);
