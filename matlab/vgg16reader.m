function img = vgg16reader(filename)
    img = imresize(imread(filename),[224,224]);
end