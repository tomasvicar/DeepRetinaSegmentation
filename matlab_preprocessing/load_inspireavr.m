function[] = load_inspireavr(rc, path, out_f)

%% INSPIRE-AVR
degree = 30;

images = dir([path 'INSPIRE-AVR\INSPIRE-AVR\org\*.jpg']);

procento=0;

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);
    
    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'inspireavr', 0);
    
    imname= [ 'inspireavr_na_na_'  in(6:end) ];
    
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
end