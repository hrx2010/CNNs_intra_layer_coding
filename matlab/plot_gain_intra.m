clear all;
close all;

archname = 'vgg16';
load(sprintf('%s_gain_intra.mat',archname),'varH','varX');

map = vega10;
map(10,:) = 0.5;
colorind = [10,2,0,3,5]

for i = 3
    for j = [1,2,4,5]
        x = numel(varH{i,j});
        semilogy(0:x-1,sort(varH{i,j}(:).*varX{i,j}(:),'asc'),...
                 'Color',map(colorind(j),:));
        hold on;
    end
    hold off;
end
xticks((0:0.25:1).*(x-1));
xticklabels(0:25:100);
ticks = xticklabels;
ticks{end} = '$\%$';
xticklabels(ticks);
axis([0,x-1,10^-2,10^+4]);

yticks(10.^(-2:2:4));
yticklabels({'$10^{-4}$','$10^{-2}$','$10^{+0}$','\textrm{Var}'});
yticklabels({'','','','',''});

grid on;
set(gca,'YMinorGrid','off');
set(gcf,'Color','none');
pdfprint(sprintf('temp_%d.pdf',1),'Width',12,'Height',9,'Position',[2.25,1.5,9.25,7]);

% archname = 'alexnet';
% imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
% labeldir = './ILSVRC2012_val.txt';
% testsize = 100;

% [neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
% [layers,lclass] = removeLastLayer(neural);
% neural = assembleNetwork(layers);

% l_kernel = findconv(neural.Layers);
% l_length = length(l_kernel);

% trannames = {'idt','dct','dst','klt_5000_intra','kkt_5000_intra'};
% t_length = length(trannames);

% gains = zeros(l_length,t_length,2);
% layers = neural.Layers(l_kernel);
% varH = cell(l_length,length(trannames));
% varX = cell(l_length,length(trannames));

% for l = 1:2%l_length
%      layer = layers(l);
%      [h,w,p,q,g] = size(layer.Weights);
%      X = activations(neural,images,neural.Layers(l_kernel(l)-1).Name);
%      X = X - mean(mean(mean(X,1),2),4); % subtract per-channel means
%      H = layer.Weights;

%      for t = 1:length(trannames)
%          T = gettrans(trannames{t},archname,l);
%          varH{l,t} = hvars2(H,T{1},h,w);
%          varX{l,t} = xvars2(X,T{3},h,w);
%          gains(l,t,1) = geomean(varX{l,t}(:),'omitnan');
%          gains(l,t,2) = geomean(varH{l,t}(:),'omitnan');
%          disp(sprintf('%s %14s | layer %03d (%5d coefficients) is %5.2f %5.2f %5.2f dB ', ...
%                       archname, trannames{t}, l, numel(H), 10*log10(gains(l,t,1)), ...
%                       10*log10(gains(l,t,2)), sum(10*log10(gains(l,t,:)))));
%      end
% end

