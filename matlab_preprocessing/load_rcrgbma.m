function[] = load_rcrgbma(rc, path, out_f)

%% RC-RGB-RA
degree = 45;

images = dir([path 'RC-RGB-MA\Original\*.jpg']);

for i=1:length(images)

    im=imread([images(i).folder '\' images(i).name ]);
           
    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'rcrgbma', 0);
    
    ind=strfind(images(i).name,'(');
    in=images(i).name(ind+1:end-5);
    imname= [ 'rcrgbma_na_na_'  in ];
    
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
end