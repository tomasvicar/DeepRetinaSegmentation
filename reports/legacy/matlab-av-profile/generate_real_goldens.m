function generate_real_goldens(output_dir, helper_dir, drive_image, drive_fov, hrf_image, hrf_fov)
% Rebuild real goldens from user-owned helpers and explicitly supplied raw files.
% MATLAB is a reference generator only; installed Python prediction needs no MATLAB.
% Public toolbox API calls only; no proprietary source inspection or copies.
if ~exist(output_dir,'dir'), mkdir(output_dir); end
addpath(helper_dir);
files={drive_image,hrf_image,drive_image};
masks={drive_fov,hrf_fov,drive_fov};
names={'drive','hrf','drive_35'};datasets={'drive','hrf','drive'};
degrees=[45,60,45];ppds=[25,25,35];
for k=1:3
 raw=imread(files{k});fov=imread(masks{k});
 if ndims(fov)==3,fov=rgb2gray(fov);end
 fov=logical(fov);
 [row,col,radius,~]=souradnice(im2double(raw),datasets{k},fov);
 [resized,~,~,~,prepared_fov]=image_adjustment(raw,ppds(k),degrees(k),0,0,0,datasets{k},fov);
 normalized=local_contrast_and_clahe(resized,prepared_fov);
 stored=uint16(round(normalized*4096));
 save(fullfile(output_dir,[names{k},'_golden.mat']),'raw','fov','row','col','radius','resized','prepared_fov','normalized','stored','-v7');
 if k<3
  mask=imerode(prepared_fov,strel('disk',1));dt=bwdist(mask==1);
  [sx,sy]=imgradientxy(dt,'sobel');
  [xx,yy]=meshgrid(1:size(dt,2),1:size(dt,1));xxx=xx-sx/4.*dt;yyy=yy-sy/4.*dt;
  save(fullfile(output_dir,[names{k},'_coords.mat']),'mask','dt','sx','sy','xxx','yyy','-v7');
  extrapolated=zeros(size(resized));
  for c=1:3,extrapolated(:,:,c)=interp2(resized(:,:,c),xxx,yyy,'linear',0);end
  sigma=80/(55/25);kernel=250/(55/25);
  G=imgaussfilt(extrapolated,sigma,'Padding','replicate');
  corrected=(extrapolated-G)./G+.5;corrected=max(0,min(1,corrected));
  hsv=rgb2hsv(corrected);v_before=hsv(:,:,3);
  tiles=round(size(v_before)/kernel);
  v_after=adapthisteq(v_before,'NumTiles',tiles,'ClipLimit',.005);
  save(fullfile(output_dir,[names{k},'25_stages.mat']),'extrapolated','G','corrected','v_before','v_after','tiles','-v7');
 end
 fprintf('%s center=%g,%g radius=%g shape=%dx%d\n',names{k},row,col,radius,size(stored,1),size(stored,2));
end
end
