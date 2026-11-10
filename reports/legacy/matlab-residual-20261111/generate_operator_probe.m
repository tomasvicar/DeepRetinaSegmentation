% Public API only; no toolbox source inspection. One bounded synthetic batch.
rng(61111,'twister');sizes=[3 5;15 19;16 19;17 19;31 35;32 35;33 35;64 19;65 19];
cases=struct('name',{},'distance',{},'gx',{},'gy',{},'gx_filter',{},'gy_filter',{});
kx=[-1 0 1;-2 0 2;-1 0 1];ky=kx';
for k=1:size(sizes,1)
 value=single(rand(sizes(k,1),sizes(k,2))*300);
 [gx,gy]=imgradientxy(value,'sobel');
 fx=imfilter(value,kx,'replicate','corr');fy=imfilter(value,ky,'replicate','corr');
 cases(end+1)=struct('name',sprintf('random_%dx%d',sizes(k,1),sizes(k,2)),...
  'distance',value,'gx',gx,'gy',gy,'gx_filter',fx,'gy_filter',fy);
 fprintf('%s imgradientxy/imfilter mismatches gx=%d gy=%d\n',cases(end).name,nnz(gx~=fx),nnz(gy~=fy));
end
matlab_version=version;computer_type=computer;
save('operator_probe.mat','cases','matlab_version','computer_type','-v7');
