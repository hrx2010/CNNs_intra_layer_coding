function net = resnet101py
    load('resnet101.mat','weight','biases','bnorm_mean','bnorm_vars');
    net = resnet101;
    net = modifyConvParams(net,weight,biases,bnorm_mean,bnorm_vars);
end