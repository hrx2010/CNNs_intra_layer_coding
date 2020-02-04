function covX = loadstatpy(archname,layers,xform)
    files = dir(sprintf('%s_%s_stats_*.mat',archname,xform));
    covX = cell(length(layers),1);
    %avgX = cell(length(layers),1);

    for l = 1:length(layers)
        %avgX{l} = 0;
        covX{l} = 0;
    end

    for f = 1:length(files)
        stats = load(sprintf('%s',files(f).name),'cov','avg');
        for l = 1:length(layers)
            covX{l} = covX{l} + double(stats.cov{l});
            %avgX{l} = avgX{l} + double(stats.avg{l});
        end
    end

    for l = 1:length(layers)
        [h,w,p,q,g] = size(layers(l).Weights);

        switch xform
          case 'joint'
            for i = 1:size(covX{l},2)
                covX{l}(:,i) = reshape(permute(reshape(covX{l}(:,i),[p,h,w]),[2,3,1]),[],1);
            end
            for i = 1:size(covX{l},1)
                covX{l}(i,:) = reshape(permute(reshape(covX{l}(i,:),[p,h,w]),[2,3,1]),[],1);
            end
        end
    end
end