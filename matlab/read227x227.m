function img = alexnetreader(filename)
    img = imresize(imread(filename),[227,227]);
    if size(img,3) == 1
        img = repmat(img,[1,1,3]);
    end
end