clear all;
close all;

archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
testsize = 128;
maxsteps = 64;
l = 2;
[neural,images] = loadnetwork(archname, imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
l_kernel = findconv(neural.Layers); 
l_length = length(l_kernel);
[h,w,p,q] = size(neural.Layers(l_kernel(l)).Weights);

kltfun = @(x) klt2(x);
fftfun = @(x) circshift(dft2(x),[3,3]);
dctfun = @(x) dct2(x);
dstfun = @(x) dst2(x);
id_fun = @(x) circshift(x,[3,3]);

funarray = {id_fun,fftfun,dctfun,dstfun};
for f = 1:length(funarray);
    close all; 
    figure(f);
    xcoeffs = funarray{f}(neural.Layers(l_kernel(l)).Weights);
    psd = mean(mean(mean(abs(xcoeffs).^2,3),4),5);
    bar3c(psd,1);
    colormap(viridis(64));
    xticks(1:5);
    yticks(1:5);
    % zticks(1:3);
    % xticklabels();
    % yticklabels();
    % zticklabels();
    % xlabel('$n$');
    % ylabel('$m$');
    axis([0.5,h+0.5,0.5,h+0.5,0,max(psd(:))],'square');
    view(-45,45);
    disp(sum(psd(:)));
    camproj('perspective');
    pdfprint(sprintf('temp_%d.pdf',f),'Width',8.40,'Height',8.40,'Position',[0.5,1,7.9,7.9]);
    input('press any key to continue...');
end

close all;
figure(1);
for f = 1:length(funarray);
    xcoeffs = funarray{f}(neural.Layers(l_kernel(l)).Weights);
    psd = mean(mean(mean(abs(xcoeffs).^2,3),4),5);
    semilogy(0:h*w-1,sort(psd(:),'descend'));
    hold on;
end
xlabel('Coefficients');
ylabel('Squared magnitude');
pdfprint('temp1.pdf','Width',21,'Height',12,'Position',[3.5,3,16.5,8]);
