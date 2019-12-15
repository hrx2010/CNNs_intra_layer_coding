function [neural,images] = loadnetwork(archname,imagedir, labeldir, testsize)
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

    labels = neural.Layers(end).Classes(textread(GetFullPath(labeldir)));
    images = imageDatastore(imagedir,'ReadFcn',readerfun,'Labels',labels);
    files = images.Files(1:testsize);
    labels = images.Labels(1:testsize);
    images.Files = files;
    images.Labels = labels;
end