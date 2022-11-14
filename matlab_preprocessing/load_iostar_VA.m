function[] = load_iostar_VA(rc, path, out_f)
degree = 45;

%% IOSTAR
% mkdir([out_f '\IOSTAR'])
% 
% images = dir([path 'IOSTAR\image\*.jpg']);
% 
% for i=1:length(images)
%     disp(['IOSTAR: ' num2str(i) '/' num2str(length(images))])
%     in=images(i).name(1:end-4);
% 
%     im=imread([images(i).folder '\' images(i).name ]);
%     ves=imread([path 'IOSTAR\GT\' in '_GT.tif']);
%     fov=imread([path 'IOSTAR\mask\' in '_Mask.tif']);
%     va_tmp=imread([path 'IOSTAR\AV_GT\' in '_AV.tif']);
%     
%     va = zeros(size(ves));
%     va(va_tmp(:,:,1)>230 & va_tmp(:,:,2)<230) = 1;
%     va(va_tmp(:,:,3)>230 & va_tmp(:,:,2)<230) = 2;
%     
%     chck_labels = unique(va(:));
%     if length(chck_labels)>3
%         disp(['IOSTAR: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
%     end
%     
%     [I,V,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'iostar',fov);
%     VA = uint8(VA).*uint8(V);
%     V(VA==0) = 0;
%     I = local_contrast_and_clahe(I,fov>0);
%     I = uint16(round(I.*2.^12));
%     
%     imname= [ 'iostar_'  in(6:7) ];
%     
%     dicomwrite(I(:,:,1),[out_f '\IOSTAR\' imname '_R.dcm'])
%     dicomwrite(I(:,:,2),[out_f '\IOSTAR\' imname '_G.dcm'])
%     dicomwrite(I(:,:,3),[out_f '\IOSTAR\' imname '_B.dcm'])
%     dicomwrite(uint16(V),[out_f '\IOSTAR\' imname '_ves.dcm'])
%     dicomwrite(uint16(fov),[out_f '\IOSTAR\' imname '_fov.dcm'])
%     dicomwrite(uint16(VA),[out_f '\IOSTAR\' imname '_va.dcm'])
% 
% end

images = dir([path 'IOSTAR\image\*.jpg']);

for i=1:length(images)
    disp(['IOSTAR: ' num2str(i) '/' num2str(length(images))])
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);
    ves=imread([path 'IOSTAR\GT\' in '_GT.tif']);
    fov=imread([path 'IOSTAR\mask\' in '_Mask.tif']);
    va_tmp=imread([path 'IOSTAR\AV_GT\' in '_AV.tif']);
    
    va = zeros(size(ves));
    va(va_tmp(:,:,1)>230 & va_tmp(:,:,2)<230) = 1;
    va(va_tmp(:,:,3)>230 & va_tmp(:,:,2)<230) = 2;
    
    chck_labels = unique(va(:));
    if length(chck_labels)>3
        disp(['IOSTAR: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
    end
    
    [I,~,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'iostar',fov);
%     VA = uint8(VA).*uint8(V);
%     V(VA==0) = 0;
    I = local_contrast_and_clahe(I,fov>0);
    I = uint8(round(I.*(2.^8-1)));
    VA = uint8(VA);
    
    imname= [ 'iostar_'  in(6:7) ];
    
    niftiwrite(cat(3,I(:,:,1), I(:,:,1)), [out_f '\imagesTr\' imname '_0000'],'Compressed',true)
    niftiwrite(cat(3,I(:,:,2), I(:,:,2)),[out_f '\imagesTr\' imname '_0001'],'Compressed',true)
    niftiwrite(cat(3,I(:,:,3), I(:,:,3)),[out_f '\imagesTr\' imname '_0002'],'Compressed',true)
    niftiwrite(cat(3,VA, VA),[out_f '\labelsTr\' imname],'Compressed',true)

end
end