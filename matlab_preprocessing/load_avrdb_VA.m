function[] = load_avrdb_VA(rc, path, out_f)
degree = 30;

%% AVRDB
% mkdir([out_f '\AVRDB'])
% images = dir([path 'AVRDB']);
% images(1:2)=[];
% 
% for i=1:length(images)
%     disp(['AVRDB: ' num2str(i) '/' num2str(length(images))])
%     im=imread([path 'AVRDB\' images(i).name '\' images(i).name '.JPG']);
%     ves=imread([path 'AVRDB\' images(i).name '\' images(i).name '--vessels.jpg']);
%     ves=ves(:,:,1)<128;
%     try
%         art=imread([path 'AVRDB\' images(i).name '\' images(i).name '--artery.jpg']);
%     end
%     try
%         art=imread([path 'AVRDB\' images(i).name '\' images(i).name '--arteries.jpg']);
%     end
%     art=mean(im2double(art),3);
%     art(art>0.80)=0;
%     art(art~=0)=1;
%     try
%         vein=imread([path 'AVRDB\' images(i).name '\' images(i).name '--vein.jpg']);
%     end
%     try
%         vein=imread([path 'AVRDB\' images(i).name '\' images(i).name '--veins.jpg']);
%     end
%     vein=mean(im2double(vein),3);
%     vein(vein>0.80)=0;
%     vein(vein~=0)=1;
%     va = art;
%     va(vein==1)=2;
%     
%     chck_labels = unique(va(:));
%     if length(chck_labels)>3
%         disp(['AVRDB: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
%     end
%     
%     [I,V,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'avrdb',0);
%     VA = uint8(VA).*uint8(V);
%     V(VA==0) = 0;
%     I = local_contrast_and_clahe(I,fov>0);
%     I = uint16(round(I.*2.^12));
% 
%     in=images(i).name;
%     imname= [ 'avrdb_'  in(end-3:end)  ];
% 
%     dicomwrite(I(:,:,1),[out_f '\AVRDB\' imname '_R.dcm'])
%     dicomwrite(I(:,:,2),[out_f '\AVRDB\' imname '_G.dcm'])
%     dicomwrite(I(:,:,3),[out_f '\AVRDB\' imname '_B.dcm'])
%     dicomwrite(uint16(V),[out_f '\AVRDB\' imname '_ves.dcm'])
%     dicomwrite(uint16(fov),[out_f '\AVRDB\' imname '_fov.dcm'])
%     dicomwrite(uint16(VA),[out_f '\AVRDB\' imname '_va.dcm'])
% 
% end

images = dir([path 'AVRDB']);
images(1:2)=[];

for i=1:length(images)
    disp(['AVRDB: ' num2str(i) '/' num2str(length(images))])
    im=imread([path 'AVRDB\' images(i).name '\' images(i).name '.JPG']);
    ves=imread([path 'AVRDB\' images(i).name '\' images(i).name '--vessels.jpg']);
    ves=ves(:,:,1)<128;
    try
        art=imread([path 'AVRDB\' images(i).name '\' images(i).name '--artery.jpg']);
    end
    try
        art=imread([path 'AVRDB\' images(i).name '\' images(i).name '--arteries.jpg']);
    end
    art=mean(im2double(art),3);
    art(art>0.80)=0;
    art(art~=0)=1;
    try
        vein=imread([path 'AVRDB\' images(i).name '\' images(i).name '--vein.jpg']);
    end
    try
        vein=imread([path 'AVRDB\' images(i).name '\' images(i).name '--veins.jpg']);
    end
    vein=mean(im2double(vein),3);
    vein(vein>0.80)=0;
    vein(vein~=0)=1;
    va = art;
    va(vein==1)=2;
    
    chck_labels = unique(va(:));
    if length(chck_labels)>3
        disp(['AVRDB: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
    end
    
    [I,~,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'avrdb',0);
%     VA = uint8(VA).*uint8(V);
%     V(VA==0) = 0;
    I = local_contrast_and_clahe(I,fov>0);
    I = uint8(round(I.*(2.^8-1)));
    VA = uint8(VA);

    if size(I,1)~=size(VA,1) || size(I,2)~=size(VA,2)
        disp(['AVRDB: ' in 'has incorrect size!'])
    end

    in=images(i).name;
    imname= [ 'avrdb_'  in(end-3:end)  ];

    niftiwrite(cat(3,I(:,:,1), I(:,:,1)), [out_f '\imagesTr\' imname '_0000'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTr\' imname '_0000']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,1), I(:,:,1)), [out_f '\imagesTr\' imname '_0000'],info,'Compressed',true)
    niftiwrite(cat(3,I(:,:,2), I(:,:,2)),[out_f '\imagesTr\' imname '_0001'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTr\' imname '_0001']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,2), I(:,:,2)), [out_f '\imagesTr\' imname '_0001'],info,'Compressed',true)
    niftiwrite(cat(3,I(:,:,3), I(:,:,3)),[out_f '\imagesTr\' imname '_0002'],'Compressed',true)
    info = niftiinfo([out_f '\imagesTr\' imname '_0002']);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,I(:,:,3), I(:,:,3)), [out_f '\imagesTr\' imname '_0002'],info,'Compressed',true)
    niftiwrite(cat(3,VA, VA),[out_f '\labelsTr\' imname],'Compressed',true)
    info = niftiinfo([out_f '\labelsTr\' imname]);
    info.PixelDimensions(3) = 999;
    info.raw.pixdim(4) = 999;
    niftiwrite(cat(3,VA, VA),[out_f '\labelsTr\' imname],info,'Compressed',true)

end
end