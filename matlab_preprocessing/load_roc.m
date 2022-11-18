function[] = load_roc(rc, path, out_f)

%% ROC
degree = 45;

images = dir([path 'ROC\images\*.jpg']);
images = [images; dir([path 'ROC\ROCtraining\*.jpg'])];

for i=1:length(images)

    im=imread([images(i).folder '\' images(i).name ]);
    
    
    [I,~,~,~, fov]=image_adjustment(im,rc,degree,0,0,0, 'roc', 0);
    
    in=images(i).name(1:end-4);

    ind=strfind(in,'_');
    imname= [ 'roc_' in(ind+1:end) '_na_'  in(6:ind)  ];

    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])

end
end