clc;clear all; close all;
data_path = '../../Sada_01';

degree = 45;
rc = 25;

files = subdir([data_path '/*_L.JPG']);
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
%     file_save = [filepath '/ImageAnalysis/VesselsSeg/' name '_preprocessed_norm' '.png'];
    file_save = [filepath '/ImageAnalysis/VesselsSeg/' name '_preprocessed' '.png'];
    
    im=imread(file);

    T1 = 3;
    T2 = 6;
    if contains(file,'Gacr_01_017_01_L')
        T1 = 2;
        T2 = 5;
    end
    if contains(file,'Gacr_01_015_01_L')
        T1 = 1;
        T2 = 3;
    end
    if contains(file,'Gacr_01_014_01_L')
        T1 = 6;
        T2 = 10;
    end

 
    tmp = rgb2gray(im2double(im))*255;
    fov = hysthresh(tmp, T1, T2, 4);
    fov = imclose(fov,strel('disk',6));
    fov = imopen(fov,strel('disk',6));
    fov = bwareafilt(fov,1);
    fov = imfill(fov,'holes');


%     figure();
%     imshow(tmp,[])
%     hold on;
%     visboundaries(fov)
%     title(num2str(file_num))
%     drawnow;


    
    im = im2double(im);
    [rad,sloup,prumer,fov]=souradnice(im, 'xxx', fov);
    lengthI =  2*prumer;
    % lengthI =  max(maxr-minr,maxc-minc);
  
    im=imresize(im,(rc*degree)/lengthI); 
    fov=imresize(fov,(rc*degree)/lengthI,'nearest'); 

    im = uint8(im*255);

%     img_interp = uint8(local_contrast_and_clahe(double(im)/255,fov>0)*255);
    img_interp = im;
    
    mkdir(fileparts(file_save))
    imwrite(img_interp,file_save);
    

end




