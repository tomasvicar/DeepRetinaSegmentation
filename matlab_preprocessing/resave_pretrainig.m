close all;clear all;clc;

% try
% poolobj = gcp('nocreate');
% delete(poolobj);
% end
% parpool()

path = 'D:/DeepRetinaSegmentation/datasets_tmp/';

resolution = 25;  % pixels/degree - e.g. 30
if resolution==25
    size_mean = [1042.86346616742,1345.84661674244];
else
   error('unknown') 
end

output_folder = '../../data_preprocessed_hdf5_tmp';



data_path = [path 'EyePACS'];
files = subdir([data_path '/*.jpeg']);

% fprintf(1,'%s\n\n',repmat('.',1,length(files)));
for file_num=54459:length(files)
% for file_num=1:length(files)
%     fprintf(1,'\b|\n');
    
    disp([num2str(file_num) '/' num2str(length(files))])
    
    
    filename = files(file_num).name;
    
    im=imread(filename);
    im=im2double(im);
    
    size_actual = [size(im,1),size(im,2)];
    
    factor =  mean(size_mean/size_actual);
    
    im = imresize(im,factor);
    
    
    [filepath,in,ext] = fileparts(filename);
    
    
    imname= [ 'EyePACS_'  in];
    
    
    try
        imwrite_single2(im,[output_folder '\Pretraining\' imname '.tiff'])
    catch EM
        
        save(['../../error' num2str(file_num) '.mat'],'EM')
        
    end

end






