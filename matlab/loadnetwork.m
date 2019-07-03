function [neural,imds] = loadnetwork(archname, filepath)
    switch archname
      case 'alexnet'
        readerfun = @read227x227;
        neural = alexnet;
      case 'vgg16'
        readerfun = @read224x224;
        neural = vgg16;
      case 'resnet50'
        readerfun = @read224x224;
        neural = resnet50;
      case 'densenet201'
        readerfun = @read224x224;
        neural = densenet201;
      case 'mobilenetv2'
        readerfun = @read224x224;
        neural = mobilenetv2;
    end

    imds = imageDatastore(filepath,'ReadFcn',readerfun);
end