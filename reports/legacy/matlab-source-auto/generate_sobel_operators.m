clear;rng(418,'twister');reference_root=getenv('DEEPRETINA_REFERENCE_GOLDENS');if isempty(reference_root),error('Set DEEPRETINA_REFERENCE_GOLDENS to Drive/HRF reference directory');end;cases=struct('name',{},'distance',{},'gx',{},'gy',{});
for k=1:2
 if k==1,path=fullfile(reference_root,'drive_golden.mat');name='drive';else,path=fullfile(reference_root,'hrf_golden.mat');name='hrf';end
 Z=load(path,'prepared_fov');mask=imerode(Z.prepared_fov,strel('disk',1));distance=bwdist(mask==1);[gx,gy]=imgradientxy(distance,'sobel');cases(end+1)=struct('name',name,'distance',distance,'gx',gx,'gy',gy);
end
[x,y]=meshgrid(1:160,1:128);masks={((x-80).^2+(y-64).^2)<54^2,((x-7).^2+(y-30).^2)<42^2,x>30&x<130&y>25&y<90,mod(x+2*y,19)<5,rand(128,160)>.8,true(128,160)};
for k=1:numel(masks)
 mask=imerode(masks{k},strel('disk',1));if ~any(mask(:)),mask(64,80)=true;end
 distance=bwdist(mask==1);[gx,gy]=imgradientxy(distance,'sobel');cases(end+1)=struct('name',sprintf('synthetic%d',k),'distance',distance,'gx',gx,'gy',gy);
end
save('sobel_operator_goldens.mat','cases','-v7');fprintf('operator cases=%d plusexistingDiaRet\n',numel(cases));
