clear;rng(11827,'twister');warning('off','backtrace');warning('off','all');
inputs={};names={};
for k=1:200
 kind=mod(k-1,5);m=480+mod(k,127);
 if kind==0,v=rand(m,1);name=sprintf('uniform_%03d',k);
 elseif kind==1
  modes=3+mod(k,6);means=sort(rand(1,modes));v=[];
  for q=1:modes,part=means(q)+(0.005+rand*.07)*randn(30+mod(q*k,110),1);v=[v;part];end
  v=min(max(v,0),1);name=sprintf('mixture_%03d',k);
 elseif kind==2
  levels=6+mod(k,27);centers=sort(rand(1,levels));v=[];
  for q=1:levels,v=[v;repmat(centers(q),1+mod(q*k,63),1)];end
  name=sprintf('sparse_levels_%03d',k);
 elseif kind==3
  v=rand(m,1).^(.25+mod(k,9)*.5);name=sprintf('power_skew_%03d',k);
 else
  counts=1+mod((1:256)*k,37);v=repelem((0:255)/255,counts)';name=sprintf('dense_histogram_%03d',k);
 end
 inputs{end+1}=v;names{end+1}=name;
end
cases=struct('name',{},'image',{},'n',{},'thresholds',{},'em',{});p=0;
for k=1:numel(inputs)
 for n=[3 4 5]
  p=p+1;[t,em]=multithresh(inputs{k},n);cases(p)=struct('name',names{k},'image',inputs{k},'n',n,'thresholds',t,'em',em);
 end
 if mod(k,25)==0,fprintf('completed_images=%d threshold_cases=%d\n',k,p);save('firstcut_holdout.mat','cases','-v7');end
end
save('firstcut_holdout.mat','cases','-v7');
