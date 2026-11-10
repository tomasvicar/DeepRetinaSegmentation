clear;Z=load('source_auto_chain.mat');c=Z.cases(3);img=c.resized;mask0=c.resized_fov;
mask=imerode(mask0,strel('disk',1));distance=bwdist(mask==1);[gx,gy]=imgradientxy(distance,'sobel');
[xx,yy]=meshgrid(1:size(distance,2),1:size(distance,1));x=xx-gx/4.*distance;y=yy-gy/4.*distance;
extrapolated(:,:,1)=interp2(img(:,:,1),x,y,'linear',0);extrapolated(:,:,2)=interp2(img(:,:,2),x,y,'linear',0);extrapolated(:,:,3)=interp2(img(:,:,3),x,y,'linear',0);
sigma=80/(55/25);background=imgaussfilt(extrapolated,sigma,'Padding','replicate');corrected=(extrapolated-background)./background+.5;corrected(corrected<0)=0;corrected(corrected>1)=1;
hsv=rgb2hsv(corrected);v_before=hsv(:,:,3);kernel=250/(55/25);v_after=adapthisteq(v_before,'NumTiles',[round(size(img,1)/kernel) round(size(img,2)/kernel)],'ClipLimit',.005);hsv(:,:,3)=v_after;normalized=hsv2rgb(hsv);normalized(repmat(mask0==0,[1 1 3]))=0;
fprintf('instrumented_vs_source_max=%.17g\n',max(abs(normalized(:)-c.normalized(:))));
save('diaret_photometry_stages.mat','distance','gx','gy','x','y','extrapolated','background','corrected','v_before','v_after','normalized','-v7');
