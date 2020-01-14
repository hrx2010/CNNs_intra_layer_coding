function img = read227x227(filename)
    img = imread(filename);
    [h,w,d] = size(img);
    newdim = [256,256]; %ceil((227/min(h,w))*[h,w]);
    img = imresize(img,newdim,'lanczos3','Antialiasing',true);
    offset = ceil((newdim - [227,227])/2);
    img = img(offset(1)+(1:227),offset(2)+(1:227),:);
    if d == 1
        img = repmat(img,[1,1,3]);
    end
end