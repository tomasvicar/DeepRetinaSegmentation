clc;clear all;close all;

data_path = 'D:\DeepRetinaSegmentation\data_preprocessed\Images';


image_names  = subdir([data_path '/*.png']);

image_names = {image_names(:).name};

sizes = zeros(length(image_names),2);

for img_num = 1:length(image_names)
   disp([num2str(img_num) '/' num2str(length(image_names))])
    
    
   image_name = image_names{img_num};
    
    
   info = imfinfo(image_name);
    
   sizes(img_num,:) = [info.Height,info.Width];
end



size_mean = mean(sizes,1);

% [1008.70406525809,1091.40024070607]