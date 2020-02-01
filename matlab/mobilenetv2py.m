function net = mobilenetv2py
    load('mobilenetv2.mat','weight','biases','bnorm_mean','bnorm_vars');
    net = mobilenetv2;
    net = modifyConvParams(net,weight,biases,bnorm_mean,bnorm_vars);
end