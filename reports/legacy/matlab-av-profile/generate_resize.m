for name={'drive','hrf'}
 n=name{1};load([n '_golden.mat'],'resized','prepared_fov');mask=imerode(prepared_fov,strel('disk',1));dt=bwdist(mask==1);[sx,sy]=imgradientxy(dt,'sobel');[xx,yy]=meshgrid(1:size(dt,2),1:size(dt,1));xxx=xx-sx/4.*dt;yyy=yy-sy/4.*dt;
 save([n '_coords.mat'],'mask','dt','sx','sy','xxx','yyy','-v7');fprintf('%s types dt=%s sx=%s xxx=%s\n',n,class(dt),class(sx),class(xxx));
end
cases=struct('name',{},'input',{},'scale',{},'output',{},'mask',{},'nearest',{});n=0;
for shape={[1 1],[2 3],[7 11],[8 10]}
 s=shape{1};I=reshape(sin(1:prod(s)),s);
 for scale=[.3,.75,1,1.3,2,3.125]
  n=n+1;cases(n)=struct('name',sprintf('%dx%d_scale%g',s(1),s(2),scale),'input',I,'scale',scale,'output',imresize(I,scale),'mask',I>0,'nearest',imresize(I>0,size(imresize(I,scale)),'nearest'));
 end
end
save('synthetic_resize.mat','cases','-v7');

levels=uint8(0:255);double_levels=im2double(levels);save('im2double_levels.mat','levels','double_levels','-v7');
