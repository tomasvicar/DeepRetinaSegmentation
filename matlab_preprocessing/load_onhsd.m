function[] = load_onhsd(rc, path, out_f)
degree = 45;

%% Onhsd

images = dir([path 'Onhsd\Images\*.bmp']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);
    center = load([path 'Onhsd\Clinicians\' in '_C.mat']);
    center = center.ONHCentre;
    boundary = load([path 'Onhsd\Clinicians\' in '_AnsuONH.mat']);
    boundary = boundary.ONHEdge;
    x = zeros(1,24);
    y = zeros(1,24);
    angles = pi/12 * [1:24];
    for i=1:24
        angle = angles(i);
        x(i) = center(1)+cos(angle)*boundary(i);
        y(i) = center(2)+sin(angle)*boundary(i);
        
    end
    disc1 = poly2mask(x,y,size(im,1),size(im,2));
    boundary = load([path 'Onhsd\Clinicians\' in '_Bob.mat']);
    boundary = boundary.ONHEdge;
    x = zeros(1,24);
    y = zeros(1,24);
    angles = pi/12 * [1:24];
    for i=1:24
        angle = angles(i);
        x(i) = center(1)+cos(angle)*boundary(i);
        y(i) = center(2)+sin(angle)*boundary(i);
        
    end
    disc2 = poly2mask(x,y,size(im,1),size(im,2));
    
    boundary = load([path 'Onhsd\Clinicians\' in '_David.mat']);
    boundary = boundary.ONHEdge;
    x = zeros(1,24);
    y = zeros(1,24);
    angles = pi/12 * [1:24];
    for i=1:24
        angle = angles(i);
        x(i) = center(1)+cos(angle)*boundary(i);
        y(i) = center(2)+sin(angle)*boundary(i);
        
    end
    disc3 = poly2mask(x,y,size(im,1),size(im,2));
    
    boundary = load([path 'Onhsd\Clinicians\' in '_Lee.mat']);
    boundary = boundary.ONHEdge;
    x = zeros(1,24);
    y = zeros(1,24);
    angles = pi/12 * [1:24];
    for i=1:24
        angle = angles(i);
        x(i) = center(1)+cos(angle)*boundary(i);
        y(i) = center(2)+sin(angle)*boundary(i);
        
    end
    disc4 = poly2mask(x,y,size(im,1),size(im,2));


    [I,D1,D2,D3, fov]=image_adjustment(im,rc,degree,disc1,disc2,disc3, 'onhsd', 0);
    [I,D4,~,~, ~]=image_adjustment(im,rc,degree,disc4, 0, 0, 'onhsd', 0);
    
    ind=strfind(in,'-');
    in(ind)=[];
    imname= [ 'onhsd_na_na_'  in  ];   
    
   
    imwrite_2_h5(I,out_f, ['\Images\' imname ])
    imwrite_2_h5(D1,out_f, ['\Disc\' imname '_disc'])
    imwrite_2_h5(D2,out_f, ['\Disc\' imname '_disc2'])
    imwrite_2_h5(D3,out_f, ['\Disc\' imname '_disc3'])
    imwrite_2_h5(D4,out_f, ['\Disc\' imname '_disc4'])
    imwrite_2_h5(fov,out_f, ['\Fov\' imname '_fov'])
    
end

end
