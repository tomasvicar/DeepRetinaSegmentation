clear;warning('off','all');[x,y]=meshgrid(1:160,1:128);
base=uint8(cat(3,80+x*.7,40+y*.9,30+mod(x+3*y,100)));
shapes={((x-80).^2+(y-64).^2)<54^2,((x-80)/72).^2+((y-64)/57).^2<1,x>8&x<153&y>8&y<121};
cases=struct('name',{},'dataset',{},'image',{},'initial_low',{},'final_low',{},'initial_closed',{},'closed',{},'filled',{},'boundary',{},'row',{},'col',{},'radius',{},'processing_fov',{},'error',{});p=0;
for k=1:3
 for background=[0 1]
  image=base;image(repmat(~shapes{k},[1 1 3]))=background;
  for d=1:3
   if d==1,dataset='stare';n=3;factor=1;bpx=.269;tolerance=.1;
   elseif d==2,dataset='diaretdb';n=4;factor=.1;bpx=.1634;tolerance=.1;
   else,dataset='drishtigs';n=5;factor=1;bpx=.1568;tolerance=.05;end
   gray=rgb2gray(im2double(image));low=min(multithresh(gray,n))*factor;initial_low=low;high=max(gray(:))*.8;
   labels=bwlabel(gray>low);selected=ismember(labels,unique(labels(gray>high)));initial_closed=imclose(selected,strel('disk',10));closed=initial_closed;
   white=nnz(closed)/numel(gray);
   if white>1-bpx+tolerance||white<1-bpx-tolerance
    [counts,edges]=histcounts(gray(:),250);index=find(cumsum(counts)>bpx*numel(gray),1);low=edges(index);
    labels=bwlabel(gray>low);selected=ismember(labels,unique(labels(gray>high)));closed=imclose(selected,strel('disk',10));
   end
   filled=imfill(closed,'holes');boundary=edge(closed);row=NaN;col=NaN;radius=NaN;processing_fov=[];err='';
   try,[row,col,radius,processing_fov]=souradnice(im2double(image),dataset,0);catch E,err=E.message;end
   p=p+1;cases(p)=struct('name',sprintf('shape%d-bg%d-%s',k,background,dataset),'dataset',dataset,'image',image,'initial_low',initial_low,'final_low',low,'initial_closed',initial_closed,'closed',closed,'filled',filled,'boundary',boundary,'row',row,'col',col,'radius',radius,'processing_fov',processing_fov,'error',err);
  end
 end
end
save('firstcut_synthetic_fov.mat','cases','-v7');fprintf('synthetic FOV cases=%d\n',p);
