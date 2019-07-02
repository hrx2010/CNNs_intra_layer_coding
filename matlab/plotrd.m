clear all;
close all;

archname = 'alexnet';
load(archname);

X = hist_coded(:,:,1);
Y = squeeze(mean(hist_Y_sse,3))/1000;

Xh = zeros(size(X))*NaN;
Yh = zeros(size(Y))*NaN;

for i = 1:size(X,2)
    x = X(:,i);
    y = Y(:,i);
    k = cleanhull(x(~isnan(x)),y(~isnan(y)));
    Xh(1:length(k),i) = x(k);
    Yh(1:length(k),i) = y(k);
end

figure(1);
i = 4;
plot(X(:,i),Y(:,i),'.','MarkerSize',8);
hold on;
plot(Xh(:,i),Yh(:,i),'.-','MarkerSize',8);
xticks(0:2:6);
yticks(0:0.02:0.08);
axis([0,6,0,0.08]);
xlabel(sprintf('Rate for slice %d (bits)',i));
ylabel('Mean squared error');
pdfprint('temp0.pdf','Width',20,'Height',12,'Position',[3.5,3,15.5,8]);

figure(2);
i = 1:64;%[0,16,32,48,64]+1
plot(Xh(:,i),Yh(:,i),'.-','MarkerSize',8);
xticks(0:2:6);
yticks(0:0.1:0.3);
axis([0,6,0,0.3]);
legend({'1','17','33','49','65'});
xlabel(sprintf('Rate (bits)'));
ylabel('Mean squared error');
pdfprint('temp1.pdf','Width',20,'Height',12,'Position',[3.5,3,15.5,8]);

figure(3);
semilogy(Xh(:,i),Yh(:,i),'.-','MarkerSize',8);
xticks(0:2:6);
%yticks(0:0.1:0.3);
axis([0,6,10^-7,10^0]);
legend({'1','17','33','49','65'});
xlabel('Rate (bits)');
ylabel('Mean squared error');
pdfprint('temp2.pdf','Width',20,'Height',12,'Position',[3.5,3,15.5,8]);