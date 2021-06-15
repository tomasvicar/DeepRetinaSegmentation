function[] = load_refuge(rc, path, out_f)
degree = 45;

%% REFUGE Training
images = dir([path 'REFUGE\Training400\Glaucoma\*.jpg']);
images = [images; dir([path 'REFUGE\Training400\Non-Glaucoma\*.jpg'])];
masks = dir([path 'REFUGE\Annotation-Training400\Disc_Cup_Masks\Glaucoma\*.bmp']);
masks = struct2table([masks; dir([path 'REFUGE\Annotation-Training400\Disc_Cup_Masks\Non-Glaucoma\*.bmp'])]); 

for i=1:length(images)
    
    in=images(i).name(1:end-4);
    radek = find(string(cell2mat(masks.name))==[in '.bmp']);
    
    im=imread([images(i).folder '\' images(i).name ]);
    mask=imread(cell2mat([table2array(masks(radek,2)) '\' table2array(masks(radek,1)) ]));
    disc=mask<150;
    cup=mask<50;

    [I,D,C,~,fov]=image_adjustment(im,rc,degree,disc,cup,0, 'refuge', 0);
    
    if in(1)=='g'
    imname= [ 'refuge_train_glaucoma_'  in  ];
    elseif in(1)=='n'
    imname= [ 'refuge_train_healthy_'  in  ];   
    end
   
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(D,[out_f '\Disc\' imname '_disc.tiff'])
    imwrite_single(C,[out_f '\Cup\' imname '_cup.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end

%% REFUGE Validation
images = dir([path 'REFUGE\REFUGE-Validation400\*.jpg']);
masks = struct2table(dir([path 'REFUGE\REFUGE-Validation400-GT\Disc_Cup_Masks\*.bmp']));

for i=1:length(images)
    
    in=images(i).name(1:end-4);
    radek = find(string(cell2mat(masks.name))==[in '.bmp']);
    
    im=imread([images(i).folder '\' images(i).name ]);
    mask=imread(cell2mat([table2array(masks(radek,2)) '\' table2array(masks(radek,1)) ]));
    disc=mask<150;
    cup=mask<50;

    [I,D,C,~, fov]=image_adjustment(im,rc,degree,disc,cup,0, 'refuge', 0);
    
    imname= [ 'refuge_validation_na_'  in  ];
       
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(D,[out_f '\Disc\' imname '_disc.tiff'])
    imwrite_single(C,[out_f '\Cup\' imname '_cup.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end

%% REFUGE Test
images = dir([path 'REFUGE\Test400\*.jpg']);
masks = dir([path 'REFUGE\REFUGE-Test-GT\Disc_Cup_Masks\G\*.bmp']);
masks = struct2table([masks; dir([path 'REFUGE\REFUGE-Test-GT\Disc_Cup_Masks\N\*.bmp'])]);

for i=1:length(images)
    
    in=images(i).name(1:end-4);
    radek = find(string(cell2mat(masks.name))==[in '.bmp']);
    
    im=imread([images(i).folder '\' images(i).name ]);
    mask=imread(cell2mat([table2array(masks(radek,2)) '\' table2array(masks(radek,1)) ]));
    disc=mask<150;
    cup=mask<50;

    [I,D,C,~, fov]=image_adjustment(im,rc,degree,disc,cup,0, 'refuge', 0);
    
    diagnose=cell2mat(table2array(masks(radek,2)));
    if diagnose(end)=='N'
    imname= [ 'refuge_test_healthy_'  in  ];
    else
     imname= [ 'refuge_test_glaucoma_'  in  ];
    end
       
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(D,[out_f '\Disc\' imname '_disc.tiff'])
    imwrite_single(C,[out_f '\Cup\' imname '_cup.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
end