function img = read224x224(filename)
    img = imread(filename);
    [h,w,d] = size(img);
    newdim = floor((256/min(h,w))*[h,w]);
    img = imresize(img,newdim,'bilinear');
    offset = ceil((newdim - [224,224])/2);
    img = img(offset(1)+(1:224),offset(2)+(1:224),:);
    if d == 1
        img = repmat(img,[1,1,3]);
    end
end