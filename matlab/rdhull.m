function k = rdhull(x,y)
    k = convhull(x(~isnan(x)),y(~isnan(y)));
    k = k([find(diff(k)<0)]);
end