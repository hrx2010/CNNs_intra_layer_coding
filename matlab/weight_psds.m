clear all;
close all;

archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
testsize = 128;
maxsteps = 64;

[net,imds] = loadnetwork(archname, filepath);
l = findconv(net.Layers); %layer

[h,w,p,q] = size(net.Layers(l).Weights);
xcoeffs = squeeze(net.Layers(l).Weights);
psd = mean(mean(abs(xcoeffs).^2,3),4);
disp(sum(psd(:)));

figure(1);
bar3c(psd,1);
colormap(viridis(64));
xticks(1:2:11);
yticks(1:2:11);
xlabel('$n$');
ylabel('$m$');
%zlabel('$\omega^2(n,m)$');
axis([0.5,h+0.5,0.5,h+0.5,0,max(psd(:))],'square');
xcoeffs = fftshift(fftshift(fft2(net.Layers(l).Weights),1),2);
xcoeffs = reshape(xcoeffs,[h*w,p*q]);
xcoeffs = reshape([sqrt(2)*imag(xcoeffs(1:(end+1)/2-1,:));
                   real(xcoeffs((end+1)/2,:));
                   sqrt(2)*real(xcoeffs((end+1)/2+1:end,:))],[h,w,p,q]);
psd = mean(mean(abs(xcoeffs).^2,3),4)*(1/h/w);
view(-45,45);
disp(sum(psd(:)));
camproj('perspective');

pdfprint('temp1.pdf','Width',21,'Height',21,'Position',[2,2,18.5,18.5]);

figure(2);
bar3c(psd,1);
colormap(viridis(64));
xticks(1:2:11);
yticks(1:2:11);
xlabel('$n$');
ylabel('$m$');
axis([0.5,h+0.5,0.5,h+0.5,0,max(psd(:))],'square');
view(-45,45);
disp(sum(psd(:)));
camproj('perspective');

pdfprint('temp2.pdf','Width',21,'Height',21,'Position',[2,2,18.5,18.5]);
