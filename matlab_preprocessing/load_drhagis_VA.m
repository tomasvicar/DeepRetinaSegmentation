function[] = load_drhagis_VA(rc, path, out_f)
degree = 45;

%% DR HAGIS
mkdir([out_f '\DRHAGIS'])
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
    fov_im_mask = repmat(fov,1,1,3);
    im(~fov_im_mask) = 0; % delete date label from images

    [I,V,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'drhagis', fov);
    
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
    
    dicomwrite(I(:,:,1),[out_f '\DRHAGIS\' imname '_R.dcm'])
    dicomwrite(I(:,:,2),[out_f '\DRHAGIS\' imname '_G.dcm'])
    dicomwrite(I(:,:,3),[out_f '\DRHAGIS\' imname '_B.dcm'])
    dicomwrite(uint16(V),[out_f '\DRHAGIS\' imname '_ves.dcm'])
    dicomwrite(uint16(fov),[out_f '\DRHAGIS\' imname '_fov.dcm'])
    dicomwrite(uint16(VA),[out_f '\DRHAGIS\' imname '_va.dcm'])

end
end