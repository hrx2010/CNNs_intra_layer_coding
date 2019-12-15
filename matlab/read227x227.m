function img = read227x227(filename)
    img = imread(filename);
    img = imresize(img,[256,256]);
    % newdim = ceil((227/min(h,w))*[h,w]);
    offset = ceil(([256,256] - [227,227])/2);
    img = img(offset(1)+(1:227),offset(2)+(1:227),:);
    [h,w,d] = size(img);
    if d == 1
        img = repmat(img,[1,1,3]);
    end
end