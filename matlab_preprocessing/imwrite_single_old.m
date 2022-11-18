function []=imwrite_single(data,name)

if islogical(data) || isa(data,'uint8')
    data = data>0;
    if length(size(data))==3
        data = data(:,:,1);
    end
    imwrite(data,name)
    return
end

data=single(data);

outputFileName = name;

t = Tiff(outputFileName,'w');

tagstruct.ImageLength     = size(data,1);
tagstruct.ImageWidth      = size(data,2);
tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample   = 32;
tagstruct.SampleFormat = 3;
tagstruct.SamplesPerPixel = size(data,3);
tagstruct.RowsPerStrip    = 16;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Software        = 'MATLAB';
tagstruct.Compression = Tiff.Compression.LZW;
t.setTag(tagstruct)


t.write(data);
t.close();



end