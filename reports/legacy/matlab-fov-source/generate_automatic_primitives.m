% Small public API oracles. No multi-threshold optimizer or real-model calls.
clear;rng(56,'twister');
basis=zeros(1,3,3);basis(1,1,1)=1;basis(1,2,2)=1;basis(1,3,3)=1;
coefficients=rgb2gray(basis);
rgb_cases=struct('name',{},'rgb',{},'gray',{});
for k=1:3
 if k==1,I=uint8(randi([0,255],[24,31,3]));name='random';
 elseif k==2,[x,y]=meshgrid(0:30,0:23);I=uint8(cat(3,8*x,10*y,4*(x+y)));name='ramps';
 else,I=uint8(ones(24,31,3)*127);name='constant';end
 rgb_cases(k)=struct('name',name,'rgb',I,'gray',rgb2gray(im2double(I)));
end
footprint=getnhood(strel('disk',10));
[x,y]=meshgrid(1:80,1:64);disk=(x-40).^2+(y-32).^2<25^2;hole=disk;hole(28:35,35:46)=false;
clipped=false(64,80);clipped(1:48,1:59)=true;clipped(20:25,20:25)=false;
masks={false(64,80),true(64,80),hole,clipped,rand(64,80)>.88};
names={'empty','full','hole','clipped','sparse_random'};
morphology=struct('name',{},'mask',{},'closed',{},'filled',{});
for k=1:numel(masks)
 closed=imclose(masks{k},strel('disk',10));filled=imfill(closed,'holes');
 morphology(k)=struct('name',names{k},'mask',masks{k},'closed',closed,'filled',filled);
end
mask=false(13,17);mask(2:5,2:5)=true;mask(6:9,6:9)=true;mask(2:5,12:15)=true;
seeds=false(size(mask));seeds(3,3)=true;
components=struct('mask',mask,'seeds',seeds,'selected',ismember(bwlabel(mask),unique(bwlabel(mask).*seeds)));
% Compute the exact source membership expression (without artificial zero seeds).
labels=bwlabel(mask);components.selected=ismember(labels,unique(labels(seeds)));
components_with_background=components;components_with_background.seeds(1,1)=true;
components_with_background.selected=ismember(labels,unique(labels(components_with_background.seeds)));
values={zeros(1,10),ones(1,10)*.37,linspace(0,1,1000),linspace(.123,.879,1000),[0,.1,.2,.3,.8,1]};
histograms=struct('values',{},'counts',{},'edges',{});
for k=1:numel(values)
 [counts,edges]=histcounts(values{k},250);
 histograms(k)=struct('values',values{k},'counts',counts,'edges',edges);
 fprintf('hist%d min=%g max=%g edge0=%.17g edgeN=%.17g bins=%d\n',k,min(values{k}),max(values{k}),edges(1),edges(end),length(counts));
end
save('automatic_primitives.mat','coefficients','rgb_cases','footprint','morphology','components','components_with_background','histograms','-v7');
fprintf('coefficients=%.17g %.17g %.17g footprint=%dx%d\n',coefficients,size(footprint,1),size(footprint,2));
