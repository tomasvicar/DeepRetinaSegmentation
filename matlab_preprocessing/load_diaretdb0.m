function[] = load_diaretdb0(rc, path, out_f)
degree = 50;

%% DiaRetDB0
images = dir([path 'DiaRetDB0\diaretdb0_v_1_1\resources\images\diaretdb0_fundus_images\*.png']);
for i=1:length(images)

    im=imread([path 'DiaRetDB0\diaretdb0_v_1_1\resources\images\diaretdb0_fundus_images\' images(i).name ]);
      
    [I,~,~,~,fov]=image_adjustment(im,rc,degree,0, 0, 0, 'diaretdb',0);
    
    in=images(i).name;
    ind=strfind(in,'.png');
    num=in(ind-3:ind-1);
    if str2num(num)>110
     imname= [ 'diaretdb0_na_healthy_'  num  ];
    else
     imname= [ 'diaretdb0_na_dr_'  num  ];
    end
    
    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])

end
end
