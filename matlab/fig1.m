clear all;
close all;

% generate per-layer curves
trans = {'kkt1','klt1','dct2','idt2'};
for j = 1:length(trans)
    for i = 1:5
        % generate_RD_frontier_intra('alexnet',trans{j},5000,i,'output',[11,1,1,1,1]); 
    end
    generate_RD_frontier_intra('alexnet',trans{j},5000,1:5,'output',[11,1,1,1,1]); 
end

trans = {'kkt','klt'};
for j = 1:length(trans)
    for i = 1:5
        % generate_RD_frontier_inter_kern('alexnet',trans{j},5000,i,'output',[3,16,16,16,16]); 
    end
    generate_RD_frontier_inter_kern('alexnet',trans{j},5000,1:5,'output',[3,16,16,16,16]); 
end