function []=imwrite_single(data,name)

if islogical(data) || isa(data,'uint8')
    
    if length(size(data))==3
        data = data(:,:,1);
    end
    
    T = mean([max(data(:)) ,min(data(:))]);
    if (T - min(data(:)))==0
       T =  Inf;
    end
    

    
    data = data>T;
    
    imwrite(uint8(data*255),replace(name,'.tiff','.png'))
    return
end

imwrite(uint8(data*255),replace(name,'.tiff','.png'))



end