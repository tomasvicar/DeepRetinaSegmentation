function[] = load_avrdb(rc, path, out_f)
degree = 30;

%% AVRDB
images = dir([path 'AVRDB']);
images(1:2)=[];

for i=1:length(images)

    im=imread([path 'AVRDB\' images(i).name '\' images(i).name '.JPG']);
    ves=imread([path 'AVRDB\' images(i).name '\' images(i).name '--vessels.jpg']);
    ves=ves(:,:,1)<128;
    figure(1)
    sgtitle(i)
    [I,V,~,~,fov]=image_adjustment(im,rc,degree,ves,0,0, 'avrdb',0);

    in=images(i).name;
    imname= [ 'avrdb_na_na_'  in(end-3:end)  ];

    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
end