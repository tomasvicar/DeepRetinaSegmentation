clear;warning('off','all');Z=load('firstcut_synthetic_fov.mat');cases=struct('name',{},'dataset',{},'image',{},'initial_low',{},'final_low',{},'initial_closed',{},'closed',{},'filled',{},'boundary',{},'row',{},'col',{},'radius',{},'processing_fov',{},'error',{});p=0;
for k=[1 16]
 image=Z.cases(k).image;gray=rgb2gray(im2double(image));
 for d=1:2
  if d==1,dataset='eophtha';bpx=.45;tolerance=.15;else,dataset='ridb';bpx=NaN;tolerance=.1;end
  low=min(multithresh(gray,5))*.5;initial_low=low;high=max(gray(:))*.8;
  labels=bwlabel(gray>low);selected=ismember(labels,unique(labels(gray>high)));initial_closed=imclose(selected,strel('disk',10));closed=initial_closed;
  white=nnz(closed)/numel(gray);if isnan(bpx),bpx=1-white;end
  if white>1-bpx+tolerance||white<1-bpx-tolerance
   [counts,edges]=histcounts(gray(:),250);index=find(cumsum(counts)>bpx*numel(gray),1);low=edges(index);
   labels=bwlabel(gray>low);selected=ismember(labels,unique(labels(gray>high)));closed=imclose(selected,strel('disk',10));
  end
  filled=imfill(closed,'holes');boundary=edge(closed);row=NaN;col=NaN;radius=NaN;processing_fov=[];err='';
  try,[row,col,radius,processing_fov]=souradnice(im2double(image),dataset,0);catch E,err=E.message;end
  p=p+1;cases(p)=struct('name',sprintf('half%d-%s',k,dataset),'dataset',dataset,'image',image,'initial_low',initial_low,'final_low',low,'initial_closed',initial_closed,'closed',closed,'filled',filled,'boundary',boundary,'row',row,'col',col,'radius',radius,'processing_fov',processing_fov,'error',err);
 end
end
save('half_factor_goldens.mat','cases','-v7');fprintf('half-factor cases=%d\n',p);
