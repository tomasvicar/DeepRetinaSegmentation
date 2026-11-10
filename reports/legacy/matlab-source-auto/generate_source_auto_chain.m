clear;warning('off','backtrace');meta=jsondecode(fileread('firstcut_real_sources.json'));S=load('firstcut_synthetic_fov.mat');
images={imread(meta(1).scratch_file),imread(meta(2).scratch_file),imread(meta(3).scratch_file),S.cases(1).image,S.cases(16).image};
datasets={'stare','drishtigs','diaretdb','stare','eophtha'};degrees=[35 30 50 12 12];
cases=struct('name',{},'dataset',{},'degree',{},'image',{},'vessel',{},'row',{},'col',{},'radius',{},'processing_fov',{},'resized',{},'resized_fov',{},'resized_vessel',{},'normalized',{},'stored_dicom12',{},'photometry_error',{});
for k=1:numel(images)
 image=images{k};dataset=datasets{k};[h,w,~]=size(image);[x,y]=meshgrid(1:w,1:h);vessel=mod(floor(x/23)+floor(y/19),2)==0;
 [row,col,radius,processing_fov]=souradnice(im2double(image),dataset,0);
 [resized,resized_vessel,~,~,resized_fov]=image_adjustment(image,25,degrees(k),vessel,0,0,dataset,0);
 normalized=[];stored_dicom12=[];err='';
 try,normalized=local_contrast_and_clahe(resized,resized_fov);stored_dicom12=uint16(round(normalized*4096));catch E,err=E.message;end
 cases(k)=struct('name',sprintf('chain%d-%s',k,dataset),'dataset',dataset,'degree',degrees(k),'image',image,'vessel',vessel,'row',row,'col',col,'radius',radius,'processing_fov',processing_fov,'resized',resized,'resized_fov',resized_fov,'resized_vessel',resized_vessel,'normalized',normalized,'stored_dicom12',stored_dicom12,'photometry_error',err);
 fprintf('%s original=%dx%d resized=%dx%d radius=%d error=%s\n',cases(k).name,h,w,size(resized,1),size(resized,2),radius,err);
end
save('source_auto_chain.mat','cases','-v7');
