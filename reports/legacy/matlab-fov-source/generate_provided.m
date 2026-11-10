% Public API/user-owned source oracle only; no proprietary toolbox source.
clear;cases=struct('name',{},'dataset',{},'mask',{},'boundary',{},'processing_fov',{},'row',{},'col',{},'radius',{},'error',{});
[x,y]=meshgrid(1:100,1:79);circle=(x-50).^2+(y-40).^2<30^2;
wire=false(79,100);wire([1,end],1:50)=true;wire(:,[1,50])=true;
clipped=wire;clipped(1,:)=false;
dense=false(79,100);dense(:,1:50)=true;
holes=circle;holes(33:45,43:56)=false;
inputs={
 'rim_loader','rimone',wire;
 'rim_clipped_wire','rimone',clipped;
 'rim_dense_source_behavior','rimone',dense;
 'rim_full_source_behavior','rimone',true(79,100);
 'rim_empty','rimone',false(79,100);
 'erlangen_full','erlangen',true(79,100);
 'erlangen_empty_source_behavior','erlangen',false(79,100);
 'erlangen_holes','erlangen',holes;
 'erlangen_video_odd','erlangen_videodata',true(80,101);
 'roc_circle','roc',circle;
 'roc_clipped','roc',((x-22).^2+(y-40).^2<35^2);
 'riga_holes','riga',holes;
 'generic_holes','drive',holes;
 'roc_empty','roc',false(79,100)
};
real=imread('real_erlangen.png');inputs(end+1,:)={'real_erlangen_source_shape','erlangen',true(size(real,1),size(real,2))};
rim=imread('real_rimone.jpg');m=size(rim,1);n=size(rim,2);frame=false(m,n);frame([1,end],1:n/2)=true;frame(:,[1,n/2])=true;
inputs(end+1,:)={'real_rimone_loader_frame','rimone',frame};
for k=1:size(inputs,1)
 name=inputs{k,1};dataset=inputs{k,2};mask=logical(inputs{k,3});
 if strcmp(dataset,'rimone'),boundary=mask;
 elseif strcmp(dataset,'erlangen')||strcmp(dataset,'erlangen_videodata'),boundary=true(size(mask));boundary(:,2:end-1)=false;
 elseif strcmp(dataset,'roc')||strcmp(dataset,'riga'),boundary=false(size(mask));boundary(20:end-20,20:end-20)=edge(mask(20:end-20,20:end-20));
 else,boundary=edge(mask);end
 raw=zeros(size(mask,1),size(mask,2),3,'uint8');if strcmp(name,'real_erlangen_source_shape'),raw=real;elseif strcmp(name,'real_rimone_loader_frame'),raw=rim;end
 row=NaN;col=NaN;radius=NaN;processing_fov=[];err='';
 try,[row,col,radius,processing_fov]=souradnice(raw,dataset,mask);catch E,err=E.message;end
 cases(k)=struct('name',name,'dataset',dataset,'mask',mask,'boundary',boundary,'processing_fov',processing_fov,'row',row,'col',col,'radius',radius,'error',err);
 fprintf('%s row=%g col=%g radius=%g boundary=%d fov=%d error=%s\n',name,row,col,radius,nnz(boundary),nnz(processing_fov),err);
end
save('provided_source_goldens.mat','cases','-v7');
