function[] = load_messidor(rc, path, out_f)
degree = 45;

%% MESSIDOR
images = dir([path 'MESSIDOR\Base11\Base11\*.tif']);
images = [images ; dir([path 'MESSIDOR\Base12\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base13\Base13\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base14\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base21\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base22\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base23\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base24\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base31\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base32\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base33\*.tif'])];
images = [images ; dir([path 'MESSIDOR\Base34\*.tif'])];


t = readtable([path 'MESSIDOR\Annotation_Base11.xls']);
t = [t; readtable([path 'MESSIDOR\Annotation_Base12.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base13.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base14.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base21.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base22.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base23.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base24.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base31.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base32.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base33.xls'])];
t = [t; readtable([path 'MESSIDOR\Annotation_Base34.xls'])];

for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([images(i).folder '\' images(i).name ]);

    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'messidor',0);
    
    ind=strfind(in,'_');
    in(ind)=[];
    tr=find(t.ImageName==string(images(i).name));
    rg=table2array(t(tr,3));
    me=table2array(t(tr,4));
    if rg>me
      imname= [ 'messidor_na_dr_'  in  ];
    elseif rg<me
      imname= [ 'messidor_na_me_'  in  ];
    elseif rg==0 || me==0
      imname= [ 'messidor_na_healthy_'  in  ];
    else
      imname= [ 'messidor_na_drme_'  in  ];
    end
        
    imwrite_2_h5(I,[out_f '\Images\' imname ])
    imwrite_2_h5(fov,[out_f '\Fov\' imname '_fov'])

end
end
