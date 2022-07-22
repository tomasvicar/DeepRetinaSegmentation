function[] = load_uoadr(rc, path, out_f)
degree = 45;

%% UoA_DR
images = dir([path 'UoA_DR\']);
dirFlags = [images.isdir]; 
dirFlags(1:2)=0;
images(~dirFlags)=[];

for i=1:length(images)
     in=images(i).name;
    
    im=imread([path 'UoA_DR\' images(i).name '\' images(i).name '.jpg']);
    if str2num(in)==72
    ves=imread([path 'UoA_DR\' images(i).name '\' images(i).name '.2.jpg']);
    disc=imread([path 'UoA_DR\' images(i).name '\' images(i).name '.1.jpg']);
    else
    ves=imread([path 'UoA_DR\' images(i).name '\' images(i).name '.1.jpg']);
    disc=imread([path 'UoA_DR\' images(i).name '\' images(i).name '.2.jpg']);
    end
    cup=imread([path 'UoA_DR\' images(i).name '\' images(i).name '.3.jpg']);

    [I,V,D,C,fov]=image_adjustment(im,rc,degree,ves,disc,cup, 'uoadr',0);
    
    if sum(str2num(in) == [1:81, 83:94, 130, 132:143, 168, 169, 171:174, 179, 193])
    imname= [ 'uoadr_na_npdr_'  in  ];
    elseif sum(str2num(in) == [82, 95:100, 131, 167, 176:178, 182:192, 194:200])
    imname= [ 'uoadr_na_pdr_'  in  ];
    elseif sum(str2num(in) == [101:129, 144:166, 170, 175, 180, 181])
    imname= [ 'uoadr_na_healthy_'  in  ];
    end
    
    I(round(0.97*(size(I,1)):end),1:round(0.25*(size(I,1))),:)=0;
    V(round(0.97*(size(I,1)):end),1:round(0.25*(size(I,1))),:)=0;
    D(round(0.97*(size(I,1)):end),1:round(0.25*(size(I,1))),:)=0;
    C(round(0.97*(size(I,1)):end),1:round(0.25*(size(I,1))),:)=0;
    fov(round(0.97*(size(I,1)):end),1:round(0.25*(size(I,1))),:)=0;
    
    imwrite_2_h5(I,[out_f '\Images\' imname ])
    imwrite_2_h5(V,[out_f '\Vessels\' imname '_ves'])
    imwrite_2_h5(D,[out_f '\Disc\' imname '_disc'])
    imwrite_2_h5(C,[out_f '\Cup\' imname '_cup'])
    imwrite_2_h5(fov,[out_f '\Fov\' imname '_fov'])

end
end
