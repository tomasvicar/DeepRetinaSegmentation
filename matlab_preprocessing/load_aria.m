function[] = load_aria(rc, path, out_f)
degree = 50;

%% ARIA HEALTHY
images = dir([path 'ARIA\aria_c_markups\aria_c_markups\*.tif']);

for i=1:length(images)

    im=imread([path 'ARIA\aria_c_markups\aria_c_markups\' images(i).name]);
    ves1=imread([path 'ARIA\aria_c_markup_vessel\aria_c_markup_vessel\' images(i).name(1:end-4) '_BDP.tif']);
    ves2=imread([path 'ARIA\aria_c_markup_vessel\aria_c_markup_vessel\' images(i).name(1:end-4) '_BSS.tif']);
    dis=imread([path 'ARIA\aria_c_markupdiscfovea\aria_c_markupdiscfovea\' images(i).name(1:end-4) '_dfs.tif']);
      
    dis=logical(im2double(dis));
    dis=imclose(dis,strel('disk',21));
    BW = imfill(dis, 'holes');
    dis = bwareafilt(BW,1);

    [I,V1,V2,D,fov]=image_adjustment(im,rc,degree,ves1,ves2,dis, 'aria',0);

    in=images(i).name;
    ind=strfind(in,'_');
    num=in(ind(2)+1:strfind(in,'.tif')-1);
    num(strfind(num,'_'))='0';
    imname= [ 'aria_na_healthy_'  num  ];

    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V1,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(V2,[out_f '\Vessels\' imname '_ves2.tiff'])
    imwrite_single(D,[out_f '\Disc\' imname '_disc.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end

%% ARIA AMD
images = dir([path 'ARIA\aria_a_markups\aria_a_markups\*.tif']);
for i=1:length(images)
    im=imread([path 'ARIA\aria_a_markups\aria_a_markups\' images(i).name]);
    ves1=imread([path 'ARIA\aria_a_markup_vessel\aria_a_markup_vessel\' images(i).name(1:end-4) '_BDP.tif']);
    ves2=imread([path 'ARIA\aria_a_markup_vessel\aria_a_markup_vessel\' images(i).name(1:end-4) '_BSS.tif']);

    [I,V1,V2,~,fov]=image_adjustment(im,rc, degree,ves1,ves2,0, 'aria',0);

    in=images(i).name;
    ind=strfind(in,'_');
    num=in(ind(2)+1:strfind(in,'.tif')-1);
    num(strfind(num,'_'))='0';
    imname= [ 'aria_na_amd_'  num  ];
    
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V1,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(V2,[out_f '\Vessels\' imname '_ves2.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])
end

%% ARIA DIABETIC
images = dir([path 'ARIA\aria_d_markups\aria_d_markups\*.tif']);

for i=1:length(images)

    im=imread([path 'ARIA\aria_d_markups\aria_d_markups\' images(i).name]);
    ves1=imread([path 'ARIA\aria_d_markup_vessel\aria_d_markup_vessel\' images(i).name(1:end-4) '_BDP.tif']);
    ves2=imread([path 'ARIA\aria_d_markup_vessel\aria_d_markup_vessel\' images(i).name(1:end-4) '_BSS.tif']);
    if i<17
    dis=imread([path 'ARIA\aria_d_markupdiscfovea\aria_d_markupdiscfovea\' images(i).name(1:end-4) '_dfs.tif']);
    elseif i==42
    dis=imread([path 'ARIA\aria_d_markupdiscfovea\aria_d_markupdiscfovea\' images(i).name(1:end-4) '.tif']);
    else
    dis=imread([path 'ARIA\aria_d_markupdiscfovea\aria_d_markupdiscfovea\' images(i).name(1:end-4) '_dfd.tif']);
    end
    
    dis=logical(im2double(dis));
    dis=imclose(dis,strel('disk',21));
    BW = imfill(dis, 'holes');
    dis = bwareafilt(BW,1);
    
    [I,V1,V2,D,fov]=image_adjustment(im,rc, degree,ves1,ves2,dis, 'aria',0);

    in=images(i).name;
    
    ind=strfind(in,'_');
    num=in(ind(2)+1:strfind(in,'.tif')-1);
    num(strfind(num,'_'))='0';
    if i<9
        num(end+1:end+4)=in(2:5);
    end
    imname= [ 'aria_na_diabetes_'  num  ];
  
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(V1,[out_f '\Vessels\' imname '_ves.tiff'])
    imwrite_single(V2,[out_f '\Vessels\' imname '_ves2.tiff'])
    imwrite_single(D,[out_f '\Disc\' imname '_disc.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
end