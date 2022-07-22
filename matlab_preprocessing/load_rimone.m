function[] = load_rimone(rc, path, out_f)
degree = 20;

%% RIM-ONE Glaucoma
images = dir([path 'RIM-ONE\RIM-ONE r3\Glaucoma and suspects\Stereo Images\*.jpg']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([images(i).folder '\' images(i).name ]);
    cup=imread([path 'RIM-ONE\RIM-ONE r3\Glaucoma and suspects\Average_masks\' in '-Cup-Avg.png' ]);
    [m,n,~]=size(im);
    fov=logical(zeros(m,n)); fov([1, end],1:n/2)=1; fov(:,[1 n/2])=1;
    [I,C,~,~, fov]=image_adjustment(im,rc,degree,cup,0,0, 'rimone', fov);
    
    in(strfind(in,'-'))=[];
    imname= [ 'rimone_na_glaucoma_'  in  ];
    
    imwrite_2_h5(I,[out_f '\Images\' imname ])
    imwrite_2_h5(C,[out_f '\Cup\' imname '_cup'])
    imwrite_2_h5(fov,[out_f '\Fov\' imname '_fov'])

end

%% RIM-ONE Healthy
images = dir([path 'RIM-ONE\RIM-ONE r3\Healthy\Stereo Images\*.jpg']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([images(i).folder '\' images(i).name ]);
    cup=imread([path 'RIM-ONE\RIM-ONE r3\Healthy\Average_masks\' in '-Cup-Avg.png' ]);
    [m,n,~]=size(im);
    fov=logical(zeros(m,n)); fov([1, end],1:n/2)=1; fov(:,[1 n/2])=1;
    [I,C,~,~, fov]=image_adjustment(im,rc,degree,cup,0,0, 'rimone', fov);
   
    in(strfind(in,'-'))=[];
    imname= [ 'rimone_na_healthy_'  in  ];
    
    imwrite_2_h5(I,[out_f '\Images\' imname ])
    imwrite_2_h5(C,[out_f '\Cup\' imname '_cup'])
    imwrite_2_h5(fov,[out_f '\Fov\' imname '_fov'])
end
end
