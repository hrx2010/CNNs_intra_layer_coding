function net = vgg16py()
    load('vgg16.mat','weight','biases');
    net = vgg16;
    net = modifyConvParams(net,weight,biases);
end