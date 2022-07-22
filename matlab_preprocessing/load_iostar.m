function[] = load_iostar(rc, path, out_f)

%% IOSTAR
degree = 45;

images = dir([path 'IOSTAR\image\*.jpg']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);
    ves=imread([path 'IOSTAR\GT\' in '_GT.tif']);
    disc=imread([path 'IOSTAR\mask_OD\' in '_ODMask.tif']);
    fov=imread([path 'IOSTAR\mask\' in '_Mask.tif']);
    
    pom=bwlabel(~disc);
    dis=pom==2;
    
    [I,V,D,~, fov]=image_adjustment(im,rc,degree,ves,dis,0, 'iostar', fov);
    
    imname= [ 'iostar_na_na_'  in(6:7) ];
    
    imwrite_2_h5(I,[out_f '\Images\' imname ])
    imwrite_2_h5(V,[out_f '\Vessels\' imname '_ves'])
    imwrite_2_h5(D,[out_f '\Disc\' imname '_disc'])
    imwrite_2_h5(fov,[out_f '\Fov\' imname '_fov'])

end
end
