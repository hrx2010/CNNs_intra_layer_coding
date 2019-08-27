clear all;
close all;

net = alexnet;
convs = [findconv(net.Layers),findfull(net.Layers)];
weights = cell(length(convs),2);

for l = 1:length(convs)
    weights{l,1} = net.Layers(convs(l)).Weights;
    weights{l,2} = net.Layers(convs(l)).Bias;
end
