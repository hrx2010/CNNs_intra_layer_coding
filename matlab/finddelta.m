function [dists,steps,coded] = finddelta(dists,steps,coded)
    [d,i] = min(dists,[],2);
    [x,~,z] = ndgrid(1:size(i,1),1:size(i,2),1:size(i,3));
    dists = squeeze(dists(sub2ind(size(dists),x,i,z)));
    steps = squeeze(steps(sub2ind(size(steps),x,i,z)));
    coded = squeeze(coded(sub2ind(size(coded),x,i,z)));
end