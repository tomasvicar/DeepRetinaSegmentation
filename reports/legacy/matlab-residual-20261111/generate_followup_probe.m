% One additional bounded public-operator batch; no toolbox source.
rng(61112,'twister');sizes=[1 1;1 19;19 1;3 5;3 19;5 19;15 5;5 15;16 5;5 16;16 16;17 17;33 35;65 19];
cases=struct('name',{},'height',{},'width',{},'input',{},'gx',{},'gy',{},'dy',{},'column_then_row_gy',{},'row_then_column_gy',{},'padded_gx',{},'padded_gy',{},'error',{});
kx=[-1 0 1;-2 0 2;-1 0 1];ky=kx';
for k=1:size(sizes,1)
 value=single(rand(sizes(k,1),sizes(k,2))*300);gx=[];gy=[];dy=[];cr=[];rc=[];px=[];py=[];error_text='';
 try
  [gx,gy]=imgradientxy(value,'sobel');dy=imfilter(value,[-1;0;1],'replicate','corr');
  cr=imfilter(dy,[1 2 1],'replicate','corr');rc=imfilter(imfilter(value,[1 2 1],'replicate','corr'),[-1;0;1],'replicate','corr');
  padded=padarray(value,[1 1],'replicate','both');px=imfilter(padded,kx,0,'corr');py=imfilter(padded,ky,0,'corr');px=px(2:end-1,2:end-1);py=py(2:end-1,2:end-1);
 catch e,error_text=e.identifier;end
 cases(end+1)=struct('name',sprintf('holdout_%dx%d',sizes(k,1),sizes(k,2)),'height',sizes(k,1),'width',sizes(k,2),'input',value,'gx',gx,'gy',gy,'dy',dy,'column_then_row_gy',cr,'row_then_column_gy',rc,'padded_gx',px,'padded_gy',py,'error',error_text);
 fprintf('%s status=%s\n',cases(end).name,error_text);
end
matlab_version=version;save('followup_probe.mat','cases','matlab_version','-v7');
