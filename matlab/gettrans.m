function T = gettrans(tranname,archname,layernum,inputnum,groupnum)
    T = cell(2,1);
    switch tranname
      case 'dct2'
        T{1} = @dct2;
        T{2} = @idct2;
        T{3} = @dct2;
        T{4} = @idct2;
      case 'dst2'
        T{1} = @dst2;
        T{2} = @idst2;
        T{3} = @dst2;
        T{4} = @idst2;
      case 'dft2'
        T{1} = @dft2;
        T{2} = @idft2;
        T{3} = @dft2;
        T{4} = @idft2;
      case 'idt2'
        T{1} = @idt2;
        T{2} = @iidt2;
        T{3} = @idt2;
        T{4} = @iidt2;
      otherwise
        load(sprintf('%s_%s',archname,tranname),'K');
        T{1} = K{layernum}{inputnum,groupnum};
        T{2} = inv(K{layernum}{inputnum,groupnum});
        T{3} = inv(K{layernum}{inputnum,groupnum}');
        T{4} = K{layernum}{inputnum,groupnum}';
    end
end
