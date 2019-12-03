function tensor = perm5(tensor,layer)
    if isa(layer,'nnet.cnn.layer.FullyConnectedLayerCustom')
        switch ndims(tensor)
          case 2
            tensor = reshape(tensor',[1,1,size(tensor')]);
          otherwise 
            tensor = squeeze(tensor)';
        end
    end
end