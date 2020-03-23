clear all;
close all; 

l = 2

load alexnetpy_klt_val_1000_8_8_output_joint_kern.mat
kltx = (1:strides(l):size(kern_Y_sse{l},3));
klty = squeeze(kern_Y_sse{l}(1,1,kltx));
semilogy(kltx-1,klty,'Color',0.50*[1,1,1]);
hold on;



load alexnetpy_ekt_val_1000_8_8_output_joint_kern.mat
ektx = (1:strides(l):size(kern_Y_sse{l},3));
ekty = squeeze(kern_Y_sse{l}(1,1,kltx));
semilogy(ektx-1,ekty,'Color','r');
hold on;

% load alexnetpy_idt_val_1000_2_2_output_inter_kern.mat
% idtx = (1:strides(l):size(kern_Y_sse{l},3));
% idty = squeeze(kern_Y_sse{l}(1,1,kltx));
% semilogy(idtx-1,idty,'Color','k');
% hold on;


xticks([0,1/3,2/3,1]*(192-16));
xticklabels([0,1/3,2/3,1]*192);
yticks(10.^[-2,-1,0,+1]);
yticklabels({'$10^{-2}$','$10^{-1}$', '$10^{+0}$','$10^{+1}$'});
%yticklabels({});
axis([0,192-16,10^-2,10^+1]);
set(gcf,'Color','none');
grid on;
set(gca,'XMinorGrid','off');
set(gca,'YMinorGrid','off');
pdfprint(sprintf('temp_%d.pdf',l),'Width',9.75,'Height',9,'Position',[2,1.25,7,7]);
    
