function[] = load_ridb(rc, path, out_f)
degree = 45;

%% RIDB

images = dir([path '\RIDB\*.jpg']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);

    [I,~,~,~,fov]=image_adjustment(im,rc,degree,0,0,0, 'ridb', 0);

    imname= [ 'ridb_na_na_'  in  ];   
 
   
    imwrite_2_h5(I,[out_f '\Images\' imname ])
    imwrite_2_h5(fov,[out_f '\Fov\' imname '_fov'])
    
end
end
