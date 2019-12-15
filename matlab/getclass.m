function Y_cats = getclass(neural,Y)
    [~,Y_cats] = max(exp(Y)./sum(exp(Y)),[],3);
    Y_cats = neural.Layers(end).Classes(Y_cats);
    Y_cats = squeeze(Y_cats);
end