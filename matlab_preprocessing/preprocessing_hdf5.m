clc;clear all; close all;

data_filename = "../../tmp_retina_hdf5.hdf5";


info = h5info(data_filename,'/Fov');

img_names = {info.Datasets(:).Name};
img_names = cellfun(@(x) replace(x,'_fov',''),img_names,UniformOutput=false);

for img_num = 3641:length(img_names)
    disp([num2str(img_num) ' / ' num2str(length(img_names))])

    img_name = img_names{img_num};

    img = h5read(data_filename,['/Images/' img_name]);

    fov = h5read(data_filename,['/Fov/' img_name '_fov']);

    img_interp = uint8(local_contrast_and_clahe(double(img)/255,fov>0)*255);


    name = ['/Images_Gauss_and_Clahe/' img_name '_gc'];

    ChunkSize = [100 100 3];
    start = [1 1 1];

    shape = size(img_interp);

    h5create(data_filename,name,shape,'Datatype','uint8','ChunkSize',ChunkSize,'Deflate',4)
    h5write(data_filename,name,img_interp,start,shape)


end