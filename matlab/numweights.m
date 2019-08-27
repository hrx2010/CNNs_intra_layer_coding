function n = numweights(layers)
    n = 0;
    convs = findconv(layers);
    for l = convs
        n = n + numel(layers(l).Weights);
    end
end