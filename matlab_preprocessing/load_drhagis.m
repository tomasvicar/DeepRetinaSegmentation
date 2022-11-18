function[] = load_drhagis(rc, path, out_f)
degree = 45;

%% DR HAGIS
images = dir([path 'DRHAGIS\DRHAGIS\Fundus_Images\*.jpg']);

for i=1:length(images)
    if str2num(images(i).name(1:end-4)) == 32
        continue
    end
    im=imread([path 'DRHAGIS\DRHAGIS\Fundus_Images\' images(i).name ]);
    ves=imread([path 'DRHAGIS\DRHAGIS\Manual_Segmentations\' images(i).name(1:end-4) '_manual_orig.png']);
    fov=imread([path 'DRHAGIS\DRHAGIS\Mask_images\' images(i).name(1:end-4) '_mask_orig.png']);

    [I,V,~,~,fov]=image_adjustment(im,rc,degree,ves,0,0, 'drhagis', fov);
    
    in=images(i).name(1:end-4);
    if sum(str2num(in) == [1:10])
    imname= [ 'drhagis_na_glaucoma_'  in  ];
    elseif sum(str2num(in) == [11:20])
    imname= [ 'drhagis_na_hypertension_'  in  ];
    elseif sum(str2num(in) == [21:23, 25:30])
    imname= [ 'drhagis_na_dr_'  in  ];
    elseif sum(str2num(in) == [31:40])
    imname= [ 'drhagis_na_amd_'  in  ];
    elseif sum(str2num(in) == [24])
    imname= [ 'drhagis_na_dramd_'  in  ];    
    end
    
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
end