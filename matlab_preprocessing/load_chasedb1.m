function[] = load_chasedb1(rc, path, out_f)
degree = 30;

%% CHASE DB1
images = dir([path 'CHASEDB1\*.jpg']);

for i=1:length(images)
    im=imread([path 'CHASEDB1\' images(i).name]);
    ves1=imread([path 'CHASEDB1\' images(i).name(1:end-4) '_1stHO.png']);
    ves2=imread([path 'CHASEDB1\' images(i).name(1:end-4) '_2ndHO.png']);

    [I,V1,V2,~,fov]=image_adjustment(im,rc,degree,ves1,ves2,0, 'chasedb1',0);
    
    in=images(i).name;
    ind=strfind(in,'_');
    num=in(ind(1)+1:strfind(in,'.jpg')-1);
    num(strfind(num,'_'))=[];
    imname= [ 'chasedb1_na_child_'  num  ];
    
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V1,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(V2,[out_f '\Vessels\' imname '_ves2.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])
end

end