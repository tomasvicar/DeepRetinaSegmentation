clc;clear all; close all;
data_path = '../../data';

% degree = 45;
% rc = 25;

video_to_25_factor = 0.4566;



files = subdir([data_path '/*.avi']);
files = {files(:).name};

% u = {};
% file_u = {};
% for file_num = 1:length(files)
%     
%     file = files{file_num};
% 
%     [filepath,name,ext] = fileparts(file);
%     
%     same = cellfun(@(x) strcmp(name,x),u,'UniformOutput',true);
%     if ~any(same)
%         u = [u,name];
%         file_u = [file_u,file];
%     end
% 
% end
% files = file_u;


for file_num = 1:length(files)
    
    file = files{file_num};




    [filepath,name,ext] = fileparts(file);


    [data,fps] = readFFFMPEGvideo(file, 'gray8', 1, 1, false);
    data = squeeze(data);

    N = size(data,3);
    for frame_num = 1:N

        im = data(:,:,frame_num);
    
        im=imresize(double(im),video_to_25_factor);
        if frame_num == 1
            data_out = zeros(size(im,1), size(im,2), 1, N,'uint8');
        end

        
%         img_interp = uint8(local_contrast_and_clahe_without_fov(im/255) * 255);
        img_interp = uint8(im);
        data_out(:,:,1,frame_num) = img_interp;

        
%         mkdir(fileparts(file_save))
%         imwrite(img_interp,file_save);

        drawnow;
    end

    writeFFFMPEGvideo([filepath, name '_preprocesed' ext], data_out, fps);
%     writeFFFMPEGvideo([filepath,'\', name '_preprocessed_norm' ext], data_out, fps);
    

end




