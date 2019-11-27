function T = gettrans(tranname,archname,layernum)
    T = cell(2,1);
    switch tranname
      case 'dct2_intra'
        T{1} = @dct2;
        T{2} = @idct2;
        T{3} = @dct2;
        T{4} = @idct2;
      case 'dct2_2'
        T{1} = @(x) dct2(x,[2,2]);
        T{2} = @(x) idct2(x,[2,2]);
        T{3} = @(x) dct2(x,[2,2]);
        T{4} = @(x) idct2(x,[2,2]);
      case 'dst2_intra'
        T{1} = @dst2;
        T{2} = @idst2;
        T{3} = @dst2;
        T{4} = @idst2;
      case 'dst2_2'
        T{1} = @(x) dst2(x,[2,2]);
        T{2} = @(x) idst2(x,[2,2]);
        T{3} = @(x) dst2(x,[2,2]);
        T{4} = @(x) idst2(x,[2,2]);
      case 'dft2_intra'
        T{1} = @dft2;
        T{2} = @idft2;
        T{3} = @dft2;
        T{4} = @idft2;
      case 'idt2_intra'
        T{1} = @idt2;
        T{2} = @iidt2;
        T{3} = @idt2;
        T{4} = @iidt2;
      case 'idt_fully'
        T{1} = 1;
        T{2} = 1;
        T{3} = 1;
        T{4} = 1;
      otherwise
        load(sprintf('%s_%s',archname,tranname),'K','invK','invKt','Kt');
        T{1} = K{layernum};
        T{2} = invK{layernum};
        T{3} = invKt{layernum};
        T{4} = Kt{layernum};
    end
end
