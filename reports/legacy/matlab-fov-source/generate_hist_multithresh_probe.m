clear;rng(401,'twister');warning('off','backtrace');
% Only public APIs and small synthetic arrays, no toolbox source inspection.
lo_values=[0 .001 .002 .003 .0039 .00392156862745098 .004 .01 .123 .17 .4 -.123];
hi_values=[.41 .57 .622636761119766 .879 .9 1];
hist_cases=struct('lo',{},'hi',{},'nbins',{},'edges',{},'same_distribution_independent',{});p=0;
for lo=lo_values
 for hi=hi_values
  if hi<=lo,continue;end
  for n=[6 10 100 249 250 251 256]
   p=p+1;[a,e]=histcounts([lo hi],n);[b,f]=histcounts(linspace(lo,hi,1000),n);
   hist_cases(p)=struct('lo',lo,'hi',hi,'nbins',n,'edges',e,'same_distribution_independent',isequal(e,f));
  end
 end
end
I={zeros(10,13),ones(10,13)*.37,repmat(linspace(0,1,256),3,1),reshape(linspace(.123,.879,1000),25,40),[ones(100,1)*.2;ones(200,1)*.8],rand(24,31),[randn(400,1)*.03+.15;randn(600,1)*.05+.45;randn(300,1)*.02+.8]};
names={'zero','constant','uniform256','range_ramp','two_levels','random','three_modes'};
multi_cases=struct('name',{},'image',{},'n',{},'thresholds',{},'em',{},'repeat_equal',{},'rng_unchanged',{},'last_warning',{});p=0;
for k=1:numel(I)
 for n=[3 4 5]
  p=p+1;lastwarn('');state=rng;[t,em]=multithresh(I{k},n);rng_unchanged=isequal(state,rng);[wm,wi]=lastwarn;
  rng(989,'twister');[repeat,~]=multithresh(I{k},n);rng(401,'twister');
  multi_cases(p)=struct('name',names{k},'image',I{k},'n',n,'thresholds',t,'em',em,'repeat_equal',isequal(t,repeat),'rng_unchanged',rng_unchanged,'last_warning',wi);
  fprintf('%s N%d t=%s em=%.17g repeat=%d rng_unchanged=%d warning=%s\n',names{k},n,mat2str(t,17),em,isequal(t,repeat),rng_unchanged,wi);
 end
end
save('hist_multi_probe.mat','hist_cases','multi_cases','-v7');
fprintf('hist_cases=%d multithresh_cases=%d\n',numel(hist_cases),numel(multi_cases));
