clear;Z=load('hist_multi_probe.mat');results=struct('name',{},'n',{},'threshold_indices',{},'counts',{},'fval',{},'exitflag',{},'iterations',{});
for k=1:numel(Z.multi_cases)
 c=Z.multi_cases(k);v=c.image;if numel(unique(v))<=c.n,continue;end
 idx=round((v-min(v(:)))/(max(v(:))-min(v(:)))*255);counts=histcounts(idx(:),-.5:1:255.5);p=counts/sum(counts);
 mass=[0 cumsum(p)];moment=[0 cumsum(p.*(0:255))];
 options=optimset('Display','off','TolX',1e-4,'TolFun',1e-4,'MaxFunEvals',200*c.n,'MaxIter',200*c.n);
 [t,fval,flag,out]=fminsearch(@(x) criterion(x,mass,moment),(1:c.n)*255/(c.n+1),options);
 results(end+1)=struct('name',c.name,'n',c.n,'threshold_indices',round(t),'counts',counts,'fval',fval,'exitflag',flag,'iterations',out.iterations);
 fprintf('%s N%d fminsearch=%s multithresh=%s\n',c.name,c.n,mat2str(round(t)),mat2str(round((c.thresholds-min(v(:)))/(max(v(:))-min(v(:)))*255)));
end
save('multithresh_fminsearch.mat','results','-v7');
function value=criterion(t,mass,moment)
 t=round(t);
 if any(t<0)||any(t>254)||any(diff(t)<=0),value=1e10;return;end
 b=[0 t+1 256]+1;m=diff(mass(b));s=diff(moment(b));
 if any(m<=0),value=1e10;return;end
 value=-sum(s.^2./m);
end
