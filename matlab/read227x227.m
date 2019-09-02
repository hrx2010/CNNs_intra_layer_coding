function img = read227x227(filename)
    img = imread(filename);
    [h,w,d] = size(img);
    newdim = floor((256/min(h,w))*[h,w]);
    offset = floor((newdim - 227)/2);
    img = imresize(img,newdim,'bilinear');
    img = img(offset(1)+(0:226),offset(2)+(0:226),:);
    if d == 1
        img = repmat(img,[1,1,3]);
    end
end