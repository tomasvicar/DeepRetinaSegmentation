function[] = load_drhagis_VA(rc, path, out_f)
degree = 45;

%% DR HAGIS
% mkdir([out_f '\DRHAGIS'])
% images = dir([path 'DRHAGIS\DRHAGIS\Fundus_Images\*.jpg']);
% 
% for i=1:length(images)
%     disp(['DRHAGIS: ' num2str(i) '/' num2str(length(images))])
%     if str2num(images(i).name(1:end-4)) == 32
%         continue
%     end
%     if str2num(images(i).name(1:end-4)) > 17 || str2num(images(i).name(1:end-4)) == 13 % protoz Fisak jeste neoznacil vsechny
%         continue
%     end
%     
%     im=imread([path 'DRHAGIS\DRHAGIS\Fundus_Images\' images(i).name ]);
%     ves=imread([path 'DRHAGIS\DRHAGIS\Manual_Segmentations\' images(i).name(1:end-4) '_manual_orig.png']);
%     fov=imread([path 'DRHAGIS\DRHAGIS\Mask_images\' images(i).name(1:end-4) '_mask_orig.png']);
%     va_name = dir([path 'DRHAGIS\DRHAGIS\Manual_Classification\' images(i).name(1:end-4) ' *.png']);
%     va = imread([va_name(1).folder '\' va_name(1).name]);
%     va(va==50) = 0;
%     va(va==100) = 1;
%     va(va==150) = 2;
%     va(va==255) = 0; % ???
%     
%     chck_labels = unique(va(:));
%     if length(chck_labels)>3
%          disp(['HRF: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
%     end
%     
%     fov_im_mask = repmat(fov,1,1,3);
%     im(~fov_im_mask) = 0; % delete date label from images
% 
%     [I,V,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'drhagis', fov);
%     VA = VA.*uint8(V);
%     V(VA==0) = 0;
%     I = local_contrast_and_clahe(I,fov>0);
%     I = uint16(round(I.*2.^12));
%     
%     in=images(i).name(1:end-4);
%     if sum(str2num(in) == [1:10])
%         imname= [ 'drhagis_glaucoma_'  in  ];
%     elseif sum(str2num(in) == [11:20])
%         imname= [ 'drhagis_hypertension_'  in  ];
%     elseif sum(str2num(in) == [21:23, 25:30])
%         imname= [ 'drhagis_dr_'  in  ];
%     elseif sum(str2num(in) == [31:40])
%         imname= [ 'drhagis_amd_'  in  ];
%     elseif sum(str2num(in) == [24])
%         imname= [ 'drhagis_dramd_'  in  ];    
%     end
%     
%     dicomwrite(I(:,:,1),[out_f '\DRHAGIS\' imname '_R.dcm'])
%     dicomwrite(I(:,:,2),[out_f '\DRHAGIS\' imname '_G.dcm'])
%     dicomwrite(I(:,:,3),[out_f '\DRHAGIS\' imname '_B.dcm'])
%     dicomwrite(uint16(V),[out_f '\DRHAGIS\' imname '_ves.dcm'])
%     dicomwrite(uint16(fov),[out_f '\DRHAGIS\' imname '_fov.dcm'])
%     dicomwrite(uint16(VA),[out_f '\DRHAGIS\' imname '_va.dcm'])
% 
% end
images = dir([path 'DRHAGIS\DRHAGIS\Fundus_Images\*.jpg']);

for i=1:length(images)
    disp(['DRHAGIS: ' num2str(i) '/' num2str(length(images))])
    if str2num(images(i).name(1:end-4)) == 32
        continue
    end
    if str2num(images(i).name(1:end-4)) > 17 || str2num(images(i).name(1:end-4)) == 13 % protoz Fisak jeste neoznacil vsechny
        continue
    end
    
    im=imread([path 'DRHAGIS\DRHAGIS\Fundus_Images\' images(i).name ]);
    ves=imread([path 'DRHAGIS\DRHAGIS\Manual_Segmentations\' images(i).name(1:end-4) '_manual_orig.png']);
    fov=imread([path 'DRHAGIS\DRHAGIS\Mask_images\' images(i).name(1:end-4) '_mask_orig.png']);
    va_name = dir([path 'DRHAGIS\DRHAGIS\Manual_Classification\' images(i).name(1:end-4) ' *.png']);
    va = imread([va_name(1).folder '\' va_name(1).name]);
    va(va==50) = 0;
    va(va==100) = 1;
    va(va==150) = 2;
    va(va==255) = 0; % ???
    
    chck_labels = unique(va(:));
    if length(chck_labels)>3
         disp(['HRF: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
    end
    
    fov_im_mask = repmat(fov,1,1,3);
    im(~fov_im_mask) = 0; % delete date label from images

    [I,~,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'drhagis', fov);
%     VA = VA.*uint8(V);
%     V(VA==0) = 0;
    I = local_contrast_and_clahe(I,fov>0);
    I = uint8(round(I.*(2.^8-1)));
    VA = uint8(VA);

    if size(I,1)~=size(VA,1) || size(I,2)~=size(VA,2)
        disp(['DRHAGIS: ' in 'has incorrect size!'])
    end
    
    in=images(i).name(1:end-4);
    if sum(str2num(in) == [1:10])
        imname= [ 'drhagis_glaucoma_'  in  ];
    elseif sum(str2num(in) == [11:20])
        imname= [ 'drhagis_hypertension_'  in  ];
    elseif sum(str2num(in) == [21:23, 25:30])
        imname= [ 'drhagis_dr_'  in  ];
    elseif sum(str2num(in) == [31:40])
        imname= [ 'drhagis_amd_'  in  ];
    elseif sum(str2num(in) == [24])
        imname= [ 'drhagis_dramd_'  in  ];    
    end
    
    niftiwrite(cat(3,I(:,:,1), I(:,:,1)), [out_f '\imagesTr\' imname '_0000'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTr\' imname '_0000']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,1), I(:,:,1)), [out_f '\imagesTr\' imname '_0000'],info,'Compressed',true)
    niftiwrite(cat(3,I(:,:,2), I(:,:,2)),[out_f '\imagesTr\' imname '_0001'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTr\' imname '_0001']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,2), I(:,:,2)), [out_f '\imagesTr\' imname '_0001'],info,'Compressed',true)
    niftiwrite(cat(3,I(:,:,3), I(:,:,3)),[out_f '\imagesTr\' imname '_0002'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTr\' imname '_0002']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,3), I(:,:,3)), [out_f '\imagesTr\' imname '_0002'],info,'Compressed',true)
    niftiwrite(cat(3,VA, VA),[out_f '\labelsTr\' imname],'Compressed',true)
    info = niftiinfo([out_f '\labelsTr\' imname]);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,VA, VA),[out_f '\labelsTr\' imname],info,'Compressed',true)
end
end