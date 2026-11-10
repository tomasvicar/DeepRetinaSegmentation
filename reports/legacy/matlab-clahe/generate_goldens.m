clear; cases=struct('name',{},'input',{},'tiles',{},'bins',{},'clip',{},'output',{}); n=0;
for value=[0,.1,.5,1]
 for bins=[8,256]
  for clip=[0,.005,.1,1]
   I=ones(16,16)*value; n=n+1;
   cases(n)=struct('name',sprintf('constant_%g_bins%d_clip%g',value,bins,clip),'input',I,'tiles',[2 2],'bins',bins,'clip',clip,'output',adapthisteq(I,'NumTiles',[2 2],'NBins',bins,'ClipLimit',clip));
  end
 end
end
for shape={[16,16],[15,19],[32,48],[8,10],[2,3]}
 s=shape{1}; I=repmat(linspace(0,1,s(2)),s(1),1);
 for bins=[8,256]
  for clip=[0,.005,.1,1]
   n=n+1;cases(n)=struct('name',sprintf('ramp_%dx%d_bins%d_clip%g',s(1),s(2),bins,clip),'input',I,'tiles',[2 2],'bins',bins,'clip',clip,'output',adapthisteq(I,'NumTiles',[2 2],'NBins',bins,'ClipLimit',clip));
  end
 end
end
rng(7,'twister');I=rand(31,47);
for tiles={[2 2],[3 4],[4 3],[8 8]}
 for bins=[8,256]
  n=n+1;t=tiles{1};cases(n)=struct('name',sprintf('random_tiles%dx%d_bins%d',t(1),t(2),bins),'input',I,'tiles',t,'bins',bins,'clip',.005,'output',adapthisteq(I,'NumTiles',t,'NBins',bins,'ClipLimit',.005));
 end
end
save('synthetic_clahe.mat','cases','-v7');
fprintf('SYNTHETIC_CASES=%d\n',n);
% HRF stage inputs come exclusively from user-owned source parameters and prior golden.
load('hrf_golden.mat','resized','prepared_fov');
mask=imerode(prepared_fov,strel('disk',1));dt=bwdist(mask==1);[sx,sy]=imgradientxy(dt,'sobel');[xx,yy]=meshgrid(1:size(dt,2),1:size(dt,1));xxx=xx-sx/4.*dt;yyy=yy-sy/4.*dt;
extrapolated=zeros(size(resized));for ch=1:3,extrapolated(:,:,ch)=interp2(resized(:,:,ch),xxx,yyy,'linear',0);end
G=imgaussfilt(extrapolated,80/(55/25),'Padding','replicate');corrected=(extrapolated-G)./G+.5;corrected=max(0,min(1,corrected));H=rgb2hsv(corrected);v_before=H(:,:,3);tiles=round(size(v_before)/(250/(55/25)));v_after=adapthisteq(v_before,'NumTiles',tiles,'ClipLimit',.005);
save('hrf25_stages.mat','extrapolated','G','corrected','v_before','v_after','tiles','-v7');
