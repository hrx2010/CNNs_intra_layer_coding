function covX = loadstatpy(archname,layers)
    files = dir([archname,'_stats_*.mat']);
    covX = cell(length(layers),1);

    for l = 1:length(layers)
        covX{l} = 0;
    end

    for f = 1:length(files)
        stats = load(sprintf('%s',files(f).name),'cov');
        for l = 1:length(layers)
            covX{l} = covX{l} + double(stats.cov{l});
        end
    end

    for l = 1:length(layers)
        sz = size(perm5(layers(l).Weights,layers(l)));
        for i = 1:size(covX{l},1)
            covX{l}(:,i) = reshape(permute(reshape(covX{l}(:,i),sz([3,1,2])),[2,3,1]),[],1);
        end
        for i = 1:size(covX{l},2)
            covX{l}(i,:) = reshape(permute(reshape(covX{l}(i,:),sz([3,1,2])),[2,3,1]),[],1);
        end
    end
end