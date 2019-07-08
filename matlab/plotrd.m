clear all;
close all;

archname = 'alexnet';
testsize = 1024;
load([archname,'_',num2str(testsize)]);

X = hist_coded(:,:,1);
Y = squeeze(mean(hist_Y_sse,3))/1000;

Xh = zeros(size(X))*NaN;
Yh = zeros(size(Y))*NaN;

for i = 1:size(X,2)
    x = X(:,i);
    y = Y(:,i);
    k = rdhull(x,y);
    Xh(1:length(k),i) = x(k);
    Yh(1:length(k),i) = y(k);
end

vega = vega10(10);

figure(1);
i = 1;
plot(X(:,i),Y(:,i),'.','Color',0.5*[1,1,1],'MarkerSize',8);
hold on;
plot(Xh(:,i),Yh(:,i),'.-','MarkerSize',8,'Color',vega(1,:));
xticks(0:2:8);
yticks(0:0.01:0.05);
axis([0,8,0,0.05]);
xlabel('Rate (bits)');
ylabel('Mean squared error');
pdfprint('temp1.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);

figure(2);
i = 1:16:96;
plot(Xh(:,i),Yh(:,i),'.-','MarkerSize',8);
xticks(0:2:8);
yticks(0:0.1:0.3);
axis([0,8,0,0.3]);
legend({'1','17','33','49','65'});
xlabel(sprintf('Rate (bits)'));
ylabel('Mean squared error');
pdfprint('temp1.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);

figure(3);
semilogy(Xh(:,i),Yh(:,i),'.-','MarkerSize',8);
xticks(0:2:8);
yticks(10.^(-8:2:0));
axis([0,8,10^-8,10^0]);
legend({'1','17','33','49','65'});
xlabel('Rate (bits)');
ylabel('Mean squared error');
pdfprint('temp2.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);