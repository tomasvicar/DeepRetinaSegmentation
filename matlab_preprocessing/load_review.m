function[] = load_review(rc, path, out_f)
%% Review HRIS
degree = 60;
images = dir([path 'Review\HRIS\*.bmp']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'review', 0);

    imname= [ 'review_na_npdr_'  in  ];   
    
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
%% Review VDIS
degree = 50;
images = dir([path 'Review\VDIS\*.bmp']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'review', 0);
    
    if sum(str2num(in(end))==[1])
    imname= [ 'review_na_a_'  in  ];   
    elseif sum(str2num(in(end))==[2])
    imname= [ 'review_na_pdr_'  in  ];
    elseif sum(str2num(in(end))==[3 4 5 8])
    imname= [ 'review_na_npdr_'  in  ]; 
    elseif sum(str2num(in(end))==[6 7])
    imname= [ 'review_na_healthy_'  in  ]; 
    end
   
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
%% Review CLRIS
degree = 50;
images = dir([path 'Review\CLRIS\*.jpg']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'review', 0);
    
    imname= [ 'review_na_a_'  in  ];   
      
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end

end