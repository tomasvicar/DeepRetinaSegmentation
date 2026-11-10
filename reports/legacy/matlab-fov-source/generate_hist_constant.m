clear;v=[0 .001 .12 .24 .25 .250001 .37 .49 .5 .74 .75 .750001 .99 1 1.24 1.25 1.37 2 3.7 37.3 -0.25 -0.37];cases=struct('value',{},'edges',{},'counts',{});
for k=1:numel(v),[c,e]=histcounts(ones(1,13)*v(k),250);cases(k)=struct('value',v(k),'edges',e,'counts',c);fprintf('const %.17g =>%.17g %.17g\n',v(k),e(1),e(end));end
save('hist_constant.mat','cases','-v7');
