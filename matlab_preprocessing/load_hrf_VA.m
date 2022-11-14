function[] = load_hrf_VA(rc, path, out_f)
degree = 60;

%% HRF
% mkdir([out_f '\HRF'])
% images = dir([path 'HRF\images\*.jpg']);
% for i=1:length(images)
%     disp(['HRF: ' num2str(i) '/' num2str(length(images))])
%     in=images(i).name(1:end-4);
%     
%     im=imread([path 'HRF\images\' images(i).name ]);
%     ves=logical(imread([path 'HRF\manual1\' in '.tif']));
%     fov=logical(rgb2gray(imread([path 'HRF\mask\' in '_mask.tif'])));
%     va = imread([path 'HRF\clasified\' in '_Eva.png']);
%     va(va==50) = 0;
%     va(va==100) = 1;
%     va(va==150) = 2;
%     va(va==255) = 0; % ???
%     
%     chck_labels = unique(va(:));
%     if length(chck_labels)>3
%         disp(['HRF: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
%     end
% 
%     [I,V,VA,~, fov]=image_adjustment(im,rc,degree,ves,va,0, 'hrf', fov);
%     VA = VA.*uint8(V);
%     V(VA==0) = 0;
%     I = local_contrast_and_clahe(I,fov>0);
%     I = uint16(round(I.*2.^12));
%     
%     ind=strfind(in,'_');
%     diagnose=in(ind(1)+1);
%     in(ind)=[];
%     if diagnose=='h'
%         imname= [ 'hrf_healthy_'  in  ];
%     elseif diagnose=='g'
%         imname= [ 'hrf_glaucoma_'  in  ];
%     elseif diagnose=='d'
%         imname= [ 'hrf_dr_'  in  ];
%     end
%     
%     dicomwrite(I(:,:,1),[out_f '\HRF\' imname '_R.dcm'])
%     dicomwrite(I(:,:,2),[out_f '\HRF\' imname '_G.dcm'])
%     dicomwrite(I(:,:,3),[out_f '\HRF\' imname '_B.dcm'])
%     dicomwrite(uint16(V),[out_f '\HRF\' imname '_ves.dcm'])
%     dicomwrite(uint16(fov),[out_f '\HRF\' imname '_fov.dcm'])
%     dicomwrite(uint16(VA),[out_f '\HRF\' imname '_va.dcm'])

images = dir([path 'HRF\images\*.jpg']);
for i=1:length(images)
    disp(['HRF: ' num2str(i) '/' num2str(length(images))])
    in=images(i).name(1:end-4);
    
    im=imread([path 'HRF\images\' images(i).name ]);
    ves=logical(imread([path 'HRF\manual1\' in '.tif']));
    fov=logical(rgb2gray(imread([path 'HRF\mask\' in '_mask.tif'])));
    va = imread([path 'HRF\clasified\' in '_Eva.png']);
    va(va==50) = 0;
    va(va==100) = 1;
    va(va==150) = 2;
    va(va==255) = 0; % ???
    
    chck_labels = unique(va(:));
    if length(chck_labels)>3
        disp(['HRF: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
    end

    [I,~,VA,~, fov]=image_adjustment(im,rc,degree,ves,va,0, 'hrf', fov);
%     VA = VA.*uint8(V);
%     V(VA==0) = 0;
    I = local_contrast_and_clahe(I,fov>0);
    I = uint8(round(I.*(2.^8-1)));
    VA = uint8(VA);
    
    ind=strfind(in,'_');
    diagnose=in(ind(1)+1);
    in(ind)=[];
    if diagnose=='h'
        imname= [ 'hrf_healthy_'  in  ];
    elseif diagnose=='g'
        imname= [ 'hrf_glaucoma_'  in  ];
    elseif diagnose=='d'
        imname= [ 'hrf_dr_'  in  ];
    end
    
    niftiwrite(cat(3,I(:,:,1), I(:,:,1)), [out_f '\imagesTr\' imname '_0000'],'Compressed',true)
    niftiwrite(cat(3,I(:,:,2), I(:,:,2)),[out_f '\imagesTr\' imname '_0001'],'Compressed',true)
    niftiwrite(cat(3,I(:,:,3), I(:,:,3)),[out_f '\imagesTr\' imname '_0002'],'Compressed',true)
    niftiwrite(cat(3,VA, VA),[out_f '\labelsTr\' imname],'Compressed',true)
end
end