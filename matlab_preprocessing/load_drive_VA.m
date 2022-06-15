function[] = load_drive_VA(rc, path, out_f)
degree = 45;

%% DRIVE/RITE Train
mkdir([out_f '\DRIVE'])

images = dir([path 'RITE\AV_groundTruth\training\images\*.tif']);
for i=1:length(images)
    disp(['DRIVE/RITE Train: ' num2str(i) '/' num2str(length(images))])
    in=images(i).name(1:end-4);
    im=imread([path 'RITE\AV_groundTruth\training\images\' images(i).name ]);
    ves=logical(mat2gray(im2double(imread([path 'RITE\AV_groundTruth\training\vessel\' in '.png']))));
    fov=logical(mat2gray(imread([path 'DRIVE\training\mask\' in '_mask.gif'])));
    va_tmp=imread([path 'RITE\AV_groundTruth\training\av\' in '.png']);
    
    va = zeros(size(ves));
    va(va_tmp(:,:,1)==255 | va_tmp(:,:,2)==255 & va_tmp(:,:,3)~=255) = 1;
    va(va_tmp(:,:,3)==255 & va_tmp(:,:,1)~=255) = 2;
    
    chck_labels = unique(va(:));
    if length(chck_labels)>3
        disp(['DRIVE: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
    end
    
    [I,V,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'drive', fov);
    VA = uint8(VA).*uint8(V);
    V(VA==0) = 0;
    I = uint16(round(I.*2.^12));
    
    num=in(1:2);
    if sum(str2num(num)==[25 26 32])
        imname= [ 'drive_train_dr_'  num ];
    else
        imname= [ 'drive_train_healthy_'  num  ];
    end

    dicomwrite(I(:,:,1),[out_f '\DRIVE\' imname '_R.dcm'])
    dicomwrite(I(:,:,2),[out_f '\DRIVE\' imname '_G.dcm'])
    dicomwrite(I(:,:,3),[out_f '\DRIVE\' imname '_B.dcm'])
    dicomwrite(uint16(V),[out_f '\DRIVE\' imname '_ves.dcm'])
    dicomwrite(uint16(fov),[out_f '\DRIVE\' imname '_fov.dcm'])
    dicomwrite(uint16(VA),[out_f '\DRIVE\' imname '_va.dcm'])
end
%% DRIVE/RITE Test
images = dir([path 'RITE\AV_groundTruth\test\images\*.tif']);
for i=1:length(images)
    disp(['DRIVE/RITE Test: ' num2str(i) '/' num2str(length(images))])
    in=images(i).name(1:end-4);
    im=imread([path 'RITE\AV_groundTruth\test\images\' images(i).name ]);
    ves=mat2gray(im2double(imread([path 'RITE\AV_groundTruth\test\vessel\' in '.png'])));
    fov=mat2gray(imread([path 'DRIVE\test\mask\' in '_mask.gif']));
    va_tmp=imread([path 'RITE\AV_groundTruth\test\av\' in '.png']);
    
    va = zeros(size(ves));
    va(va_tmp(:,:,1)>192 & va_tmp(:,:,2)<250) = 1;
    va(va_tmp(:,:,3)>192 & va_tmp(:,:,2)<250) = 2;
    
    [I,V,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'drive', fov);
    VA = uint8(VA).*uint8(V);
    V(VA==0) = 0;
    I = uint16(round(I.*2.^12));
    
    num=in(1:2);
    if sum(str2num(num)==[3 8 14 17])
        imname= [ 'drive_test_dr_'  num ];
    else
        imname= [ 'drive_test_healthy_'  num  ];
    end

    dicomwrite(I(:,:,1),[out_f '\DRIVE\' imname '_R.dcm'])
    dicomwrite(I(:,:,2),[out_f '\DRIVE\' imname '_G.dcm'])
    dicomwrite(I(:,:,3),[out_f '\DRIVE\' imname '_B.dcm'])
    dicomwrite(uint16(V),[out_f '\DRIVE\' imname '_ves.dcm'])
    dicomwrite(uint16(fov),[out_f '\DRIVE\' imname '_fov.dcm'])
    dicomwrite(uint16(VA),[out_f '\DRIVE\' imname '_va.dcm'])
end
end