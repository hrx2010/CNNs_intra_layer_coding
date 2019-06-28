function img = vgg16reader(filename)
    img = imresize(imread(filename),[224,224]);
    if size(img,3) == 1
        img = repmat(img,[1,1,3]);
    end
end