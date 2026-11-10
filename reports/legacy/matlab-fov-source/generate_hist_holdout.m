clear;rng(814,'twister');cases=struct('lo',{},'hi',{},'edges',{},'values',{},'counts',{});
for k=1:320
 if k<=280
  lo=rand*.9;hi=lo+rand*(1-lo);
 elseif k<=300
  lo=0;hi=rand;
 else
  lo=(rand-.5)*2;hi=lo+10^(rand*6-5);
 end
 values=[lo,hi,lo+(hi-lo)*rand(1,127)];[counts,edges]=histcounts(values,250);
 cases(k)=struct('lo',lo,'hi',hi,'edges',edges,'values',values,'counts',counts);
end
save('hist_holdout.mat','cases','-v7');fprintf('hist heldout synthetic cases=%d\n',numel(cases));
