clear all;
close all;

archname = 'alexnet';
load(archname);

X = hist_coded(:,:,1);
Y = squeeze(mean(hist_Y_sse,3))/1000;
D = hist_delta(:,:,1);

Xh = zeros(size(X))*NaN;
Yh = zeros(size(Y))*NaN;
Dh = zeros(size(D))*NaN;

for i = 1:size(X,2)
    x = X(:,i);
    y = Y(:,i);
    d = D(:,i);
    k = cleanhull(x(~isnan(x)),y(~isnan(y)));
    Xh(1:length(k),i) = x(k);
    Yh(1:length(k),i) = y(k);
    Dh(1:length(k),i) = d(k);
end
Lh = -diff(Yh,1)./diff(Xh,1);

figure(1);
i = 1:8:96;
semilogy(Xh(1:end-1,i),Lh(:,i),'.-','MarkerSize',8);
xlabel('Rate (bits)');
ylabel('R-D trade-off ($\lambda$)');
axis([0,6,10^-8,10^0]);
pdfprint('temp1.pdf','Width',20,'Height',12,'Position',[3.5,3,15.5,8]);

figure(2);
i = 1:8:96;
loglog(Dh(1:end-1,i),Lh(:,i),'.-','MarkerSize',8);
xlabel('Quantization step-size');
ylabel('R-D trade-off ($\lambda$)');
axis([0,2,10^-8,10^0]);
pdfprint('temp1.pdf','Width',20,'Height',12,'Position',[3.5,3,15.5,8]);

%Burrows-wheeler.
