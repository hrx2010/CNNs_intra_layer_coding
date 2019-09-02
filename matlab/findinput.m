function inputname = findinput(neural,outputname)
    inputname = '';
    for i = 2:length(neural.Layers)
        if strcmp(outputname,neural.Layers(i).Name)
            inputname = neural.Layers(i-1).Name;
            break
        end
    end
end