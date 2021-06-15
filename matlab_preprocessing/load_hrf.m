function[] = load_hrf(rc, path, out_f)
degree = 60;

%% HRF
images = dir([path 'HRF\images\*.jpg']);
for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([path 'HRF\images\' images(i).name ]);
    ves=imread([path 'HRF\manual1\' in '.tif']);
    fov=logical(rgb2gray(imread([path 'HRF\mask\' in '_mask.tif'])));

    [I,V,~,~, fov]=image_adjustment(im,rc,degree,ves,0,0, 'hrf', fov);
    
    ind=strfind(in,'_');
    diagnose=in(ind(1)+1);
    in(ind)=[];
    if diagnose=='h'
    imname= [ 'hrf_na_healthy_'  in  ];
    elseif diagnose=='g'
    imname= [ 'hrf_na_glaucoma_'  in  ];
    elseif diagnose=='d'
    imname= [ 'hrf_na_dr_'  in  ];
    end
    
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])


end
end