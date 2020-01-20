function K = gettrans(tranname,archname,layernum)
    T = cell(2,1);
    switch tranname
      case 'dct'
        K{1} = @dct2;
        K{2} = @idct2;
        K{3} = @dct2;
        K{4} = @idct2;
      case 'dct2_2'
        K{1} = @(x) dct2(x,[2,2]);
        K{2} = @(x) idct2(x,[2,2]);
        K{3} = @(x) dct2(x,[2,2]);
        K{4} = @(x) idct2(x,[2,2]);
      case 'dst'
        K{1} = @dst2;
        K{2} = @idst2;
        K{3} = @dst2;
        K{4} = @idst2;
      case 'dst'
        K{1} = @(x) dst2(x,[2,2]);
        K{2} = @(x) idst2(x,[2,2]);
        K{3} = @(x) dst2(x,[2,2]);
        K{4} = @(x) idst2(x,[2,2]);
      case 'dft'
        K{1} = @dft2;
        K{2} = @idft2;
        K{3} = @dft2;
        K{4} = @idft2;
      case 'idt'
        K{1} = @idt2;
        K{2} = @iidt2;
        K{3} = @idt2;
        K{4} = @iidt2;
      otherwise
        load(sprintf('%s_%s',archname,tranname),'T');
        K = T{layernum};
        % p = size(T{layernum},1);
        % K{1} = T{layernum}(:,:,:,1);
        % K{2} = T{layernum}(:,:,:,2);
        % K{3} = reshape(permute(reshape(T{layernum}(:,:,:,2),p,p,[]),[2,1,3]),p,[],1);
        % K{4} = reshape(permute(reshape(T{layernum}(:,:,:,1),p,p,[]),[2,1,3]),p,[],1);
    end
end
