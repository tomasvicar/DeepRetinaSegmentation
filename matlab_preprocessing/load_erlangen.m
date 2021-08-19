function[] = load_erlangen(rc, path, out_f)

%% Erlangen
degree = 20;

images = dir([path 'Erlangen\FundusImages\*.jpg']);
vessels = dir([path 'Erlangen\Segmentation_Blood_Vessels\Manually_corrected\*.tif']);
for i=1:length(vessels)
    temp=vessels(i).name;
    vess(i,1)=string([temp(1:15) temp(19)]);
end

for i=1:length(images)
    
    in=images(i).name(1:end-4);

    if ~isempty(strfind(in,'FOV'))
        continue
    end

    im=imread([images(i).folder '\' images(i).name ]);
    if logical(sum(in==vess))
        ind=find(in==vess);
        temp=imread([vessels(ind).folder '\' vessels(ind).name ]);
        ves=imresize(im2double(temp(:,:,2)),size(im(:,:,1)));

    else
        ves=0;
    end
    fov=ones(size(im,1),size(im,2));

    [I,ves,~,~, fov]=image_adjustment(im,rc,degree,ves,0,0, 'erlangen', fov);
    
    imname= [ 'erlangen_na_na_'  in(6:end) ];
    
    imwrite(I,[out_f '\Images\' imname '.png'])
    imwrite(fov,[out_f '\Fov\' imname '_fov.png'])
    if length(ves)>5
       imwrite(ves,[out_f '\Vessels\' imname '_ves.png'])
    end
end
end