function tensor = perm5(tensor,layer,inputs)
    if isa(layer,'nnet.cnn.layer.FullyConnectedLayerCustom') ...
           | isa(layer,'nnet.cnn.layer.FullyConnectedLayer')
        switch ndims(tensor)
          case 2
            [q,p,w,h,g] = size(tensor);
            if nargin == 3
                h = h * sqrt(p/inputs);
                w = w * sqrt(p/inputs);
                p = inputs;
            end
            tensor = reshape(tensor',[h,w,p,q,g]);
          otherwise 
            [h,w,p,q,g] = size(tensor);
            tensor = reshape(tensor,h*w*p,q,g);
            tensor = permute(tensor,[2,1,3]);
        end
    end
end