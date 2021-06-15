close all;clear all;clc;

path = 'G:/Sdílené disky/Retina GAČR/Data/databases/';

resolution = 25;  % pixels/degree - e.g. 30
if resolution==25
    size_mean = [1008.7,1091.4];
else
   error('unknown') 
end

output_folder = '../../data_preprocessed';

if ~exist([output_folder '\Pretraining'], 'dir')
    mkdir([output_folder '\Pretraining'])
end


data_path = [path 'EyePACS'];
files = subdir([data_path '/*.jpeg']);


for file_num=1:length(files)
    
    filename = files(file_num).name;
    
    im=imread(filename);
    im=im2double(im);
    
    size_actual = [size(im,1),size(im,2)];
    
    factor =  mean(size_mean/size_actual);
    
    im = imresize(im,factor);
    
    
    [filepath,in,ext] = fileparts(filename);
    
    
    imname= [ 'EyePACS'  in];
    
    
    
    imwrite_single(I,[output_folder '\Images\' imname '.tiff'])
    

end






