function[] = load_drive(rc, path, out_f)
degree = 45;

%% DRIVE Train
images = dir([path 'DRIVE\training\images\*.bmp']);
for i=1:length(images)
    in=images(i).name(1:end-4);
    im=imread([path 'DRIVE\training\images\' images(i).name ]);
    ves=mat2gray(im2double(imread([path 'DRIVE\training\1st_manual\' in(1:2) '_manual1.gif'])));
    fov=imread([path 'DRIVE\training\mask\' in '_mask.gif']);
    
    [I,V,~,~,fov]=image_adjustment(im,rc,degree,ves,0,0, 'drive', fov);
    
    num=in(1:2);
    if sum(str2num(num)==[25 26 32])
    imname= [ 'drive_train_dr_'  num ];
    else
    imname= [ 'drive_train_healthy_'  num  ];
    end

    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])
end
%% DRIVE Test
images = dir([path 'DRIVE\test\images\*.bmp']);
for i=1:length(images)
    in=images(i).name(1:end-4);
    im=imread([path 'DRIVE\test\images\' images(i).name ]);
    ves=mat2gray(im2double(imread([path 'DRIVE\test\1st_manual\' in(1:2) '_manual1.gif'])));
    ves2=mat2gray(im2double(imread([path 'DRIVE\test\2nd_manual\' in(1:2) '_manual2.gif'])));
    fov=imread([path 'DRIVE\test\mask\' in '_mask.gif']);

    [I,V,V2,~,fov]=image_adjustment(im,rc,degree,ves,ves2,0, 'drive', fov);
    
    num=in(1:2);
    if sum(str2num(num)==[3 8 14 17])
    imname= [ 'drive_test_dr_'  num  ];
    else
    imname= [ 'drive_test_healthy_'  num  ];
    end

    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(V2,[out_f '\Vessels\' imname '_ves2.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])
end
end