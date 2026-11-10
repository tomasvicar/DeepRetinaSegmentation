clear;rng(725,'twister');warning('off','backtrace');hist_special=struct('values',{},'edges',{},'counts',{});
values={zeros(1,12),ones(1,12)*.37,ones(1,12),ones(1,12)*2,ones(1,12)*.001,ones(1,12)*-.37,ones(1,12)*-1,[.4 .65],[.4 .6499999999999999]};
for k=1:numel(values),[c,e]=histcounts(values{k},250);hist_special(k)=struct('values',values{k},'edges',e,'counts',c);end
images={};names={};
for k=1:6
 images{end+1}=rand(19+k,23+k);names{end+1}=sprintf('uniform_seed725_%d',k);
 images{end+1}=[randn(200,1)*(.02+k*.005)+.15;randn(350,1)*.045+.42;randn(150,1)*.02+.83];names{end+1}=sprintf('three_modes_seed725_%d',k);
end
E=imread('real_erlangen.png');R=imread('real_rimone.jpg');images{end+1}=rgb2gray(im2double(E(1:8:end,1:8:end,:)));names{end+1}='real_erlangen_subsample8';images{end+1}=rgb2gray(im2double(R(1:8:end,1:8:end,:)));names{end+1}='real_rimone_subsample8';
cases=struct('name',{},'image',{},'n',{},'thresholds',{},'em',{},'histogram_edges',{},'histogram_counts',{});p=0;
for k=1:numel(images)
 [hc,he]=histcounts(images{k}(:),250);
 for n=[3 4 5]
  p=p+1;[t,em]=multithresh(images{k},n);cases(p)=struct('name',names{k},'image',images{k},'n',n,'thresholds',t,'em',em,'histogram_edges',he,'histogram_counts',hc);
  fprintf('%s N%d t=%s em=%.17g\n',names{k},n,mat2str(t,17),em);
 end
end
save('threshold_holdout.mat','hist_special','cases','-v7');fprintf('holdout threshold cases=%d\n',numel(cases));
