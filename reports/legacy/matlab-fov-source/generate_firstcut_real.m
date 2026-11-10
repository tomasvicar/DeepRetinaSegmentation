clear;warning('off','backtrace');meta=jsondecode(fileread('firstcut_real_sources.json'));
cases=struct('dataset',{},'image',{},'n',{},'factor',{},'thresholds',{},'initial_low',{},'final_low',{},'high',{},'fallback_used',{},'initial_closed',{},'closed',{},'filled',{},'boundary',{},'processing_fov',{},'row',{},'col',{},'radius',{},'error',{});
for k=1:numel(meta)
 dataset=meta(k).dataset;image=imread(meta(k).scratch_file);gray=rgb2gray(im2double(image));
 if strcmp(dataset,'stare'),n=3;factor=1;bpx=.269;tolerance=.1;
 elseif strcmp(dataset,'diaretdb'),n=4;factor=.1;bpx=.1634;tolerance=.1;
 else,n=5;factor=1;bpx=.1568;tolerance=.05;if size(image,2)>2200,bpx=.2929;end;end
 thresholds=multithresh(gray,n);low=min(thresholds)*factor;initial_low=low;high=max(gray(:))*.8;
 labels=bwlabel(gray>low);selected=ismember(labels,unique(labels(gray>high)));initial_closed=imclose(selected,strel('disk',10));closed=initial_closed;
 white=nnz(initial_closed)/numel(gray);fallback=white>1-bpx+tolerance||white<1-bpx-tolerance;
 if fallback
  [counts,edges]=histcounts(gray(:),250);index=find(cumsum(counts)>bpx*numel(gray),1);low=edges(index);
  labels=bwlabel(gray>low);selected=ismember(labels,unique(labels(gray>high)));closed=imclose(selected,strel('disk',10));
 end
 filled=imfill(closed,'holes');boundary=edge(closed);processing_fov=[];row=NaN;col=NaN;radius=NaN;err='';
 try,[row,col,radius,processing_fov]=souradnice(im2double(image),dataset,0);catch E,err=E.message;end
 cases(k)=struct('dataset',dataset,'image',image,'n',n,'factor',factor,'thresholds',thresholds,'initial_low',initial_low,'final_low',low,'high',high,'fallback_used',fallback,'initial_closed',initial_closed,'closed',closed,'filled',filled,'boundary',boundary,'processing_fov',processing_fov,'row',row,'col',col,'radius',radius,'error',err);
 fprintf('%s N%d initial_low=%.17g final_low=%.17g fallback=%d center=%g,%g radius=%g error=%s\n',dataset,n,initial_low,low,fallback,row,col,radius,err);
end
save('firstcut_real_goldens.mat','cases','-v7');
