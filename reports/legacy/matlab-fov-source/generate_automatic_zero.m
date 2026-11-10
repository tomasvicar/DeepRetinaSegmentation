% Instrument user-owned zero-low-threshold path and compare its original return.
clear;[x,y]=meshgrid(1:160,1:128);disk=(x-80).^2+(y-64).^2<54^2;
base=uint8(cat(3,170+30*y/128,80+100*x/160,60+20*sin(x/7)));
scene=base;scene(repmat(~disk,[1 1 3]))=0;
noise=scene;noise(repmat(~disk,[1 1 3]))=1;
outer=false(size(disk));outer(10:118,10:150)=true;hole=(x-80).^2+(y-64).^2<20^2;
with_hole=base;with_hole(repmat(~outer|hole,[1 1 3]))=0;
inputs={'drive_circle','drive',scene;'drive_nonzero_background','drive',noise;'drive_large_hole','drive',with_hole;'g1020_noise_fallback','g1020',noise;'g1020_dark_fallback','g1020',scene;'erlangen_auto','erlangen',scene;'rimone_auto','rimone',scene};
cases=struct('name',{},'dataset',{},'rgb',{},'gray',{},'selected',{},'initial_closed',{},'closed',{},'filled',{},'low',{},'high',{},'fallback',{},'row',{},'col',{},'radius',{},'processing_fov',{},'histogram_edges',{},'histogram_counts',{},'error',{});
for k=1:size(inputs,1)
 name=inputs{k,1};dataset=inputs{k,2};I=inputs{k,3};gray=im2double(rgb2gray(im2double(I)));
 low=0;high=max(gray(:))*.8;labels=bwlabel(gray>low);selected=ismember(labels,unique(labels(gray>high)));
 initial_closed=imclose(selected,strel('disk',10));bpx=1-nnz(initial_closed)/numel(gray);fraction=nnz(initial_closed)/numel(gray);
 if strcmp(dataset,'g1020'),bpx=.1343;end
 [oracle_counts,oracle_edges]=histcounts(gray(:),250);
 fallback=fraction>1-bpx+.1||fraction<1-bpx-.1;closed=initial_closed;
 if fallback
  [counts,edges]=histcounts(gray(:),250);index=find(cumsum(counts)>bpx*numel(gray),1);low=edges(index);
  labels=bwlabel(gray>low);selected=ismember(labels,unique(labels(gray>high)));closed=imclose(selected,strel('disk',10));
 end
 filled=imfill(closed,'holes');row=NaN;col=NaN;radius=NaN;processing_fov=[];err='';
 try,[row,col,radius,processing_fov]=souradnice(im2double(I),dataset,0);catch E,err=E.message;end
 cases(k)=struct('name',name,'dataset',dataset,'rgb',I,'gray',gray,'selected',selected,'initial_closed',initial_closed,'closed',closed,'filled',filled,'low',low,'high',high,'fallback',fallback,'row',row,'col',col,'radius',radius,'processing_fov',processing_fov,'histogram_edges',oracle_edges,'histogram_counts',oracle_counts,'error',err);
 fprintf('%s fallback=%d low=%.17g row=%g col=%g radius=%g error=%s\n',name,fallback,low,row,col,radius,err);
end
save('automatic_zero_goldens.mat','cases','-v7');
