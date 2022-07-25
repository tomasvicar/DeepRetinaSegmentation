function[] = load_eophtha(rc, path, out_f)
degree = 40;

%% E-Ophtha EX
folders = dir([path 'E-Ophtha\e_optha_EX\EX\']);
images = [];

for i=1:length(folders)
    d=dir([folders(i).folder '\' folders(i).name '\*.jpg']);
    if length(d)>0
    images=[images; d];
    end
end
for i=1:length(images)
    
    in=images(i).name(1:end-4);
    im=imread([images(i).folder '\' images(i).name ]);
    
    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'eophtha', 0);
    
    imname= [ 'eophtha_na_ex_'  in  ];

    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])
end

%% E-Ophtha MA
folders = dir([path 'E-Ophtha\e_optha_MA\MA\']);
images = [];
for i=1:length(folders)
    d=dir([folders(i).folder '\' folders(i).name '\*.jpg']);
    if length(d)>0
    images=[images; d];
    end
end
for i=1:length(images)
    
    in=images(i).name(1:end-4);
    im=imread([images(i).folder '\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'eophtha', 0);
    
    imname= [ 'eophtha_na_ma_'  in  ];


    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])
end
%% E-Ophtha healthy
folders = dir([path 'E-Ophtha\e_optha_EX\healthy\']);
folders = [folders; dir([path 'E-Ophtha\e_optha_MA\healthy\'] )];
images = [];
for i=1:length(folders)
    d=dir([folders(i).folder '\' folders(i).name '\*.jpg']);
    if length(d)>0
    images=[images; d];
    end
end
for i=1:length(images)
    
    in=images(i).name(1:end-4);
    im=imread([images(i).folder '\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'eophtha', 0);
    
    imname= [ 'eophtha_na_healthy_'  in  ];


    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])
end
end
