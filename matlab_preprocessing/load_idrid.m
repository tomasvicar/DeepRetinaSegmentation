function[] = load_idrid(rc, path, out_f)
degree = 50;

%% IDRiD Train
images = dir([path 'IDRiD\C. Localization\1. Original Images\a. Training Set\*.jpg']);
for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([path 'IDRiD\C. Localization\1. Original Images\a. Training Set\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'idrid', 0);
    
    ind=strfind(in,'_');
    num=in(ind+1:end);
    imname= [ 'idrid_na_na_ctra'  num  ];
        
    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])

end

%% IDRiD Test
images = dir([path 'IDRiD\C. Localization\1. Original Images\b. Testing Set\*.jpg']);
for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([path 'IDRiD\C. Localization\1. Original Images\b. Testing Set\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'idrid', 0);
    
    ind=strfind(in,'_');
    num=in(ind+1:end);
    imname= [ 'idrid_na_na_ctes'  num  ];
        
    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])

end
end
