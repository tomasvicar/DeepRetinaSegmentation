% Black-box public API calls plus the user's original souradnice.m; no toolbox source.
clear;load('fov_inputs.mat','inputs');
cases=struct('name',{},'mask',{},'edges',{},'unthinned',{},'threshold',{},'gx',{},'gy',{},'center_row',{},'center_col',{},'radius',{},'error',{});
for k=1:numel(inputs)
 fov=logical(inputs(k).mask);[e,t,gx,gy]=edge(fov);u=edge(fov,'sobel',[],'both','nothinning');
 row=NaN;col=NaN;radius=NaN;err='';
 try,[row,col,radius]=souradnice(zeros(size(fov,1),size(fov,2),3,'uint8'),'drive',fov);catch E,err=E.message;end
 cases(k)=struct('name',inputs(k).name,'mask',fov,'edges',e,'unthinned',u,'threshold',t,'gx',gx,'gy',gy,'center_row',row,'center_col',col,'radius',radius,'error',err);
 fprintf('%s edge=%d threshold=%.17g row=%g col=%g radius=%g error=%s\n',inputs(k).name,nnz(e),t,row,col,radius,err);
end
save('fov_synthetic_goldens.mat','cases','-v7');
for name={'drive','hrf'}
 load([name{1},'_golden.mat'],'fov','raw');
 [e,t,gx,gy]=edge(fov);u=edge(fov,'sobel',[],'both','nothinning');
 [row,col,radius]=souradnice(raw,name{1},fov);
 save([name{1},'_fov_goldens.mat'],'fov','e','t','gx','gy','u','row','col','radius','-v7');
 fprintf('%s edge=%d threshold=%.17g row=%g col=%g radius=%g\n',name{1},nnz(e),t,row,col,radius);
end
