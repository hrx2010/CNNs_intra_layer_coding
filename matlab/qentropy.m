function bits = qentropy(symbols)
    [~,~,symbols] = unique(symbols(:));
    counts = accumarray(symbols(:),1);
    counts = counts/sum(counts);
    bits = max(0,-length(symbols(:))*sum(counts.*log2(counts)));
end