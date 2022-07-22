function[] = load_vicavr(rc, path, out_f)

%% VICAVR
degree = 45;

images = dir([path 'VICAVR\images\original\*.jpg']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);
    
    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'vicavr', 0);
    
    imname= [ 'vicavr_na_na_'  in(6:end) ];
    
    imwrite_2_h5(I,[out_f '\Images\' imname ])
    imwrite_2_h5(fov,[out_f '\Fov\' imname '_fov'])

end
end
