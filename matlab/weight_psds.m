clear all;
close all;

archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
testsize = 128;
maxsteps = 64;

[net,imds] = loadnetwork(archname, filepath);
l = findconv(net.Layers); %layer
[h,w,p,q] = size(net.Layers(l).Weights);

kltfun = @(x) klt2(x);
fftfun = @(x) dft2(x);
dctfun = @(x) fct2(x);
id_fun = @(x) x;

funarray = {id_fun,kltfun,fftfun,dctfun};
for f = 1:length(funarray);
    close all; 
    figure(f);
    xcoeffs = funarray{f}(net.Layers(l).Weights);
    psd = mean(mean(abs(xcoeffs).^2,3),4);
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
    pdfprint(sprintf('temp_%d.pdf',f),'Width',21,'Height',21,'Position',[2,2,18.5,18.5]);
    input('press any key to continue...');
end

close all;
figure(1);
for f = 1:length(funarray);
    xcoeffs = funarray{f}(net.Layers(l).Weights);
    psd = mean(mean(abs(xcoeffs).^2,3),4);
    semilogy(0:h*w-1,sort(psd(:),'descend'));
    hold on;
end
xlabel('Coefficients');
ylabel('Squared magnitude');
pdfprint('temp1.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);
