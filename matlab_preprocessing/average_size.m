clc;clear all;close all;

data_path = 'D:\DeepRetinaSegmentation\data_preprocessed_hdf5\dataset.hdf5';


% image_names  = subdir([data_path '/*.png']);

info = h5info(data_path,'/Images');


image_names = {info.Datasets(:).Name};

sizes = zeros(length(image_names),2);

for img_num = 1:length(image_names)
   disp([num2str(img_num) '/' num2str(length(image_names))])
    
    
   image_name = image_names{img_num};
    
    
   info = h5info(data_path,['/Images/' image_name]);
    
   xxx = info.Dataspace.Size;
   sizes(img_num,:) = xxx(1:2);
end



size_mean = mean(sizes,1);

% [1042.86346616742,1345.84661674244]