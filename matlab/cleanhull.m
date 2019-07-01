function k = cleanhull(x,y)
    k = convhull(x,y);
    k = k([find(diff(k)<0)]);
end