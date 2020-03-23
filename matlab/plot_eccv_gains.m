clear all;
close all;

net = alexnetpy();
load('../python/alexnetpy_inter_stats_10000_1.mat');

load('./alexnetpy_klt_5_inter.mat');
layers = findconv(net.Layers,{'conv'});
% layer 2

gains = zeros(length(layers),1);

for l = 1:6%length(layers)
    G = cov{l};
    G = G./max(G(:));
    W = net.Layers(layers(l)).Weights;
    W = permute(W,[3,1,2,4])./max(W(:));
    W = W(:,:);
    U = T{l};

    Dklt = sort(diag(U(:,:,1)*(W*W')*U(:,:,2)).*diag(U(:,:,1)*G*U(:,:,2)),'desc');
    Didt = sort(diag(W*W').*diag(G),'desc');
    gains(l) = geomean(Didt)/geomean(Dklt);
end

semilogy(1:6,gains);
