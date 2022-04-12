function[] = load_iostar_VA(rc, path, out_f)

%% IOSTAR
degree = 45;

images = dir([path 'IOSTAR\image\*.jpg']);

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    im=imread([images(i).folder '\' images(i).name ]);
    ves=imread([path 'IOSTAR\GT\' in '_GT.tif']);
    fov=imread([path 'IOSTAR\mask\' in '_Mask.tif']);
    va_tmp=imread([path 'IOSTAR\AV_GT\' in '_AV.tif']);
    va = zeros(size(ves));
    va(va_tmp(:,:,1)>192 & va_tmp(:,:,2)<250) = 1;
    va(va_tmp(:,:,3)>192 & va_tmp(:,:,2)<250) = 2;
    
    [I,V,VA,~,fov]=image_adjustment(im,rc,degree,ves,va,0, 'iostar',fov);
    I = uint16(round(I.*2.^12));
    
    imname= [ 'iostar_'  in(6:7) ];
    
    dicomwrite(I(:,:,1),[out_f '\' imname '_R.dcm'])
    dicomwrite(I(:,:,2),[out_f '\' imname '_G.dcm'])
    dicomwrite(I(:,:,3),[out_f '\' imname '_B.dcm'])
    dicomwrite(uint16(V),[out_f '\' imname '_ves.dcm'])
    dicomwrite(uint16(fov),[out_f '\' imname '_fov.dcm'])
    dicomwrite(uint16(VA),[out_f '\' imname '_va.dcm'])

end
end