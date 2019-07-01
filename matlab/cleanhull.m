function k = cleanhull(k)
    k = k([find(diff(k)<0)]);
end