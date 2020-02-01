function [neural,images] = loadnetwork(archname,imagedir, labeldir, testsize)
    switch archname
      case 'alexnet'
        readerfun = @read227x227;
        neural = alexnet;
      case 'alexnetpy'
        readerfun = @read224x224py;
        neural = alexnetpy;
      case 'vgg16'
        readerfun = @read224x224;
        neural = vgg16;
      case 'vgg16py'
        readerfun = @read224x224py;
        neural = vgg16py;
      case 'resnet50'
        readerfun = @read224x224;
        neural = resnet50;
      case 'resnet50py'
        readerfun = @read224x224py;
        neural = resnet50py;
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