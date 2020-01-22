function net = alexnetpy()
    load('alexnetv2.mat','weight','biases');
    net = alexnetv2;
    net = modifyConvParams(net,weight,biases);
end