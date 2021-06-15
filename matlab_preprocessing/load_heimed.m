function[] = load_heimed(rc, path, out_f)
degree = 45;

%% HEI-MED
images = dir([path 'HEI-MED\HEI-MED-master\DMED\*.jpg']);
for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([path 'HEI-MED\HEI-MED-master\DMED\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'heimed', 0);
    
    ind=strfind(in,'_');
    in(ind)=[];
    ind=strfind(in,'(');
    in(ind)=[];
    ind=strfind(in,')');
    in(ind)=[];
    ind=strfind(in,'.dcm');
    in(ind)=[];
    imname= [ 'heimed_na_na_'  in  ];

    
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])


end
end