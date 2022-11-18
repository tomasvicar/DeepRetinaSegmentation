function[] = load_fire(rc, path, out_f)
degree = 45;

%% FIRE
images = dir([path 'FIRE\FIRE\Images\*.jpg']);
fov_o = imread([path 'FIRE\FIRE\Masks\mask.png']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([path 'FIRE\FIRE\Images\' images(i).name ]);

    [I,~,~,~,fov]=image_adjustment(im,rc,degree,0,0,0, 'fire', fov_o);
    
    ind=strfind(in,'_');
    in(ind)=[];
    imname= [ 'fire_na_na_'  in  ];


    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end

end