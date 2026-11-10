% User-owned exact image_adjustment + separately requested photometry oracle.
clear;cases=struct('name',{},'dataset',{},'input_sha256_note',{},'shape',{},'degree',{},'pixels_per_degree',{},'row',{},'col',{},'radius',{},'resized',{},'resized_fov',{},'resized_pattern',{},'normalized',{},'photometry_error',{});
files={'real_erlangen.png','erlangen';'real_rimone.jpg','rimone'};
for k=1:2
 I=imread(files{k,1});dataset=files{k,2};[h,w,~]=size(I);fov=true(h,w);
 if strcmp(dataset,'rimone'),fov=false(h,w);fov([1,end],1:w/2)=1;fov(:,[1,w/2])=1;end
 [row,col,radius,processing]=souradnice(im2double(I),dataset,fov);
 [x,y]=meshgrid(1:w,1:h);pattern=uint8(mod(floor(x/37)+floor(y/29),3));
 [resized,resized_pattern,~,~,resized_fov]=image_adjustment(I,25,20,pattern,0,0,dataset,fov);
 normalized=[];err='';try,normalized=local_contrast_and_clahe(resized,resized_fov);catch E,err=E.message;end
 cases(k)=struct('name',files{k,1},'dataset',dataset,'input_sha256_note','See real_source.json / real_rimone_source.json','shape',size(I),'degree',20,'pixels_per_degree',25,'row',row,'col',col,'radius',radius,'resized',resized,'resized_fov',resized_fov,'resized_pattern',resized_pattern,'normalized',normalized,'photometry_error',err);
 fprintf('%s input=%dx%d prepared=%dx%d center=%d,%d radius=%d processing_pixels=%d photometry_error=%s\n',dataset,h,w,size(resized,1),size(resized,2),row,col,radius,nnz(resized_fov),err);
end
save('adjustment_real_goldens.mat','cases','-v7');
