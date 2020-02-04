function [dists,steps,coded] = finddelta(dists,steps,coded)
    [d,i] = min(dists,[],2);
    [x,~,z,y] = ndgrid(1:size(i,1),1:size(i,2),1:size(i,3),1:size(i,4));
    dists = permute(dists(sub2ind(size(dists),x,i,z,y)),[1,3,4,2]);
    steps = permute(steps(sub2ind(size(steps),x,i,z,y)),[1,3,4,2]);
    coded = permute(coded(sub2ind(size(coded),x,i,z,y)),[1,3,4,2]);
end