function T = gettrans(tranname,archname,layernum)
    T = cell(2,1);
    switch tranname
      case 'klt2'
        load(sprintf('%s_%s',archname,tranname),'K');
        T{1} = K{layernum,1};
        T{2} = K{layernum,2};
      case 'dct2'
        T{1} = @dct2;
        T{2} = @idct2;
      case 'dst2'
        T{1} = @dst2;
        T{2} = @idst2;
      case 'dft2'
        T{1} = @dft2;
        T{2} = @idft2;
      case 'idt2'
        T{1} = @idt2;
        T{2} = @iidt2;
    end
end
