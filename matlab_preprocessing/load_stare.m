function[] = load_stare(rc, path, out_f)

%% STARE
degree = 35;

images = dir([path 'STARE\all-images\*.ppm']);
masks = [dir([path 'STARE\labels-ah\*.ppm']);dir([path 'STARE\labels-vk\*.ppm'])];
for k=1:length(masks)
    pom=(masks(k).name);
    masks_n(k,1)=string(pom(1:6));
end

for i=1:length(images)

    im=imread([images(i).folder '\' images(i).name ]);
    
    in=images(i).name(1:end-4);
    shoda=find(masks_n==in);
    if length(shoda)==2
        ves1=imread([masks(shoda(1)).folder '\' masks(shoda(1)).name ]);
        ves2=imread([masks(shoda(2)).folder '\' masks(shoda(2)).name ]);
    elseif length(shoda)==1
        ves1=imread([masks(shoda).folder '\' masks(shoda).name ]);
        ves2=0;
    else
        ves1=0;
        ves2=0;
    end
    
    [I,V1,V2,~, fov]=image_adjustment(im,rc,degree,ves1,ves2,0, 'stare', 0);
    

    imname= [ 'stare_na_na_'  in ];

    if length(shoda)==2
    imwrite_2_h5(V1,[out_f '\Vessels\' imname '_ves'])
    imwrite_2_h5(V2,[out_f '\Vessels\' imname '_ves2'])
    elseif length(shoda)==1
    imwrite_2_h5(V1,[out_f '\Vessels\' imname '_ves'])
    end
    imwrite_2_h5(I,[out_f '\Images\' imname ])
    imwrite_2_h5(fov,[out_f '\Fov\' imname '_fov'])

end
end
