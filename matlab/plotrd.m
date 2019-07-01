clear all;
close all;

archname = 'alexnet';
load(archname);

figure(1);
plot(hist_coded(:,:,1),-10*log10(squeeze(mean(hist_Y_sse,3))));
xticks(0:2:8);
yticks(-50:50:100)
xlabel('Rate (bits)');
ylabel('SNR (dB)');
axis([0,8,-50,100]);
%pdfprint('temp.pdf','Width',20,'Height',12,'Position',[3.5,3,15.5,8]);

figure(2);
line_Y_sse = huberfit(hist_coded(:,:,1),mean(hist_Y_sse,3));
plot(hist_coded(:,:,1),-line_Y_sse);
xticks(0:2:8);
yticks(-50:50:100)
xlabel('Rate (bits)');
ylabel('SNR (dB)');
axis([0,8,-50,100]);
%pdfprint('temp2.pdf','Width',20,'Height',12,'Position',[3.5,3,15.5,8]);
