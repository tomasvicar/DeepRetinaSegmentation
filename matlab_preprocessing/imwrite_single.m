function []=imwrite_single(data,name)

if islogical(data) || isa(data,'uint8')
    data = data>0;
    if length(size(data))==3
        data = data(:,:,1);
    end
    imwrite(uint8(data*255),replace(name,'.tiff','.png'))
    return
end

imwrite(uint8(data*255),replace(name,'.tiff','.png'))



end