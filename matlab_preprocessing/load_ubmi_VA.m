function[] = load_ubmi_VA(rc, path, out_f)
degree = 45;

files = dir([path 'Gacr*']);

% files = subdir([path '/*_L.JPG']);
files = {files(:).name};

% mkdir([out_f '\UBMI'])
% for file_num = 1:length(files)
%     disp(['UBMI: ' num2str(file_num) '/' num2str(length(files))])
%     file = files{file_num};
%     [~,imname,~] = fileparts(file);
%     
%     if ~exist([path imname '\' imname(1:end-5) 'L.jpg'],'file')
%         continue
%     end
%     
%     im=imread([path imname '\' imname(1:end-5) 'L.jpg']);
% 
%     T1 = 3;
%     T2 = 6;
%     if contains(file,'Gacr_01_017_01_L')
%         T1 = 2;
%         T2 = 5;
%     end
%     if contains(file,'Gacr_01_015_01_L')
%         T1 = 1;
%         T2 = 3;
%     end
%     if contains(file,'Gacr_01_014_01_L')
%         T1 = 6;
%         T2 = 10;
%     end
%  
%     tmp = rgb2gray(im2double(im))*255;
%     fov = hysthresh(tmp, T1, T2, 4);
%     fov = imclose(fov,strel('disk',6));
%     fov = imopen(fov,strel('disk',6));
%     fov = bwareafilt(fov,1);
%     fov = imfill(fov,'holes');
% 
%     file_seg = dir([path imname '\ImageAnalysis\VesselsSeg\*whole.png']);
%     ves=logical(mat2gray(im2double(imread([file_seg.folder '\' file_seg.name]))));
%     
%     [I,V,~,~,fov]=image_adjustment(im,rc,degree,ves,0,0, 'ubmi', fov);
%     
%     I = local_contrast_and_clahe(I,fov>0);
%     I = uint16(round(I.*2.^12));
%     
%     im_size_orig = size(im);
%     writematrix(im_size_orig, [out_f '\UBMI\' imname '_orig_size_info.csv'])
%     
%     dicomwrite(I(:,:,1),[out_f '\UBMI\' imname '_R.dcm'])
%     dicomwrite(I(:,:,2),[out_f '\UBMI\' imname '_G.dcm'])
%     dicomwrite(I(:,:,3),[out_f '\UBMI\' imname '_B.dcm'])
%     dicomwrite(uint16(V),[out_f '\UBMI\' imname '_ves.dcm'])
%     dicomwrite(uint16(fov),[out_f '\UBMI\' imname '_fov.dcm'])

for file_num = 1:length(files)
    disp(['UBMI: ' num2str(file_num) '/' num2str(length(files))])
    file = files{file_num};
    [~,imname,~] = fileparts(file);
    
    if ~exist([path imname '\' imname(1:end-5) 'L.jpg'],'file')
        continue
    end
    
    im=imread([path imname '\' imname(1:end-5) 'L.jpg']);

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

    file_seg = dir([path imname '\ImageAnalysis\VesselsSeg\*whole.png']);
    ves=logical(mat2gray(im2double(imread([file_seg.folder '\' file_seg.name]))));
    
    [I,~,~,~,fov]=image_adjustment(im,rc,degree,ves,0,0, 'ubmi', fov);
    
    I = local_contrast_and_clahe(I,fov>0);
    I = uint8(round(I.*(2.^8-1)));
    
    im_size_orig = size(im);
    writematrix(im_size_orig, [out_f '\UBMI\' imname '_orig_size_info.csv'])
    
    niftiwrite(cat(3,I(:,:,1), I(:,:,1)), [out_f '\imagesTs\' imname '_0000'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTs\' imname '_0000']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,1), I(:,:,1)), [out_f '\imagesTs\' imname '_0000'],info,'Compressed',true)

    niftiwrite(cat(3,I(:,:,2), I(:,:,2)),[out_f '\imagesTs\' imname '_0001'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTs\' imname '_0001']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,2), I(:,:,2)), [out_f '\imagesTs\' imname '_0001'],info,'Compressed',true)

    niftiwrite(cat(3,I(:,:,3), I(:,:,3)),[out_f '\imagesTs\' imname '_0002'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTs\' imname '_0002']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,3), I(:,:,3)), [out_f '\imagesTs\' imname '_0002'],info,'Compressed',true)
    disp([info.raw.dim])
end