function[] = load_diaretdb1(rc, path, out_f)
degree = 50;

%% DiaRetDB1
images = dir([path 'DiaRetDB1\diaretdb1_v_1_1\resources\images\ddb1_fundusimages\*.png']);
for i=1:length(images)

    im=imread([path 'DiaRetDB1\diaretdb1_v_1_1\resources\images\ddb1_fundusimages\' images(i).name ]);
      
    [I,~,~,~,fov]=image_adjustment(im,rc,degree,0, 0, 0, 'diaretdb',0);
    
    in=images(i).name;
    ind=strfind(in,'.png');
    num=in(ind-3:ind-1);
    
    if sum(str2num(num)==[49 57 60 62 72])
     imname= [ 'diaretdb1_na_healthy_'  num  ];
    else
     imname= [ 'diaretdb1_na_npdr_'  num  ];
    end
    
    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])

end
end
