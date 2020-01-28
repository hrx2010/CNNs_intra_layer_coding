function img = read224x224py(filename)
    rgb_avg = permute([0.485, 0.456, 0.406]',[3,2,1]);
    rgb_std = permute([0.229, 0.224, 0.225]',[3,2,1]);

    img = im2single(imread(filename));
    [h,w,d] = size(img);
    newdim = floor((256/min(h,w))*[h,w]);
    img = imresize(img,newdim,'bilinear');%,'lanczos3','Antialiasing',false);
    offset = ceil((newdim - [224,224])/2);
    img = img(offset(1)+(1:224),offset(2)+(1:224),:);
    if d == 1
        img = repmat(img,[1,1,3]);
    end
    img = (img - rgb_avg)./rgb_std;
end
