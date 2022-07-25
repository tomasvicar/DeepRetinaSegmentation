function[] = load_drishtigs(rc, path, out_f)
degree = 30;

%% Drihsti-GS Train
images = dir([path 'Drishti-GS\Training\Images\*.png']);
for i=1:length(images)
    in=images(i).name(1:end-4);
    im=imread([path 'Drishti-GS\Training\Images\' images(i).name ]);
    cup=imread([path 'Drishti-GS\Training\GT\' in '\SoftMap\' in '_cupsegSoftmap.png']);
    disc=imread([path 'Drishti-GS\Training\GT\' in '\SoftMap\' in '_ODsegSoftmap.png']);

    [I,C,D,~,fov]=image_adjustment(im,rc,degree,cup,disc,0, 'drishtigs', 0);
    
    ind=strfind(in,'_');
    num=in(ind+1:end);
    imname= [ 'drishti_train_na_'  num  ];

    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(C,out_f, ['\Cup\' imname '_cup'])
    imwrite_2_h5(D,out_f, ['\Disc\' imname '_disc'])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])
end

%% Drihsti-GS Test
images = dir([path 'Drishti-GS\Test\Images\*.png']);
for i=1:length(images)
    in=images(i).name(1:end-4);
    im=imread([path 'Drishti-GS\Test\Images\' images(i).name ]);
    cup=imread([path 'Drishti-GS\Test\Test_GT\' in '\SoftMap\' in '_cupsegSoftmap.png']);
    disc=imread([path 'Drishti-GS\Test\Test_GT\' in '\SoftMap\' in '_ODsegSoftmap.png']);
  
    [I,C,D,~,fov]=image_adjustment(im,rc,degree,cup,disc,0, 'drishtigs', 0);
    
    ind=strfind(in,'_');
    num=in(ind+1:end);
    imname= [ 'drishti_test_na_'  num  ];

    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(C,out_f, ['\Cup\' imname '_cup'])
    imwrite_2_h5(D,out_f, ['\Disc\' imname '_disc'])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])
end
end
