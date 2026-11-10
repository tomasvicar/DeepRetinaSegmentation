clear;cases=struct('name',{},'input',{},'fov',{},'normalized',{},'stored',{});
[x,y]=meshgrid(1:240,1:200);
for k=1:3
 if k==1,I=ones(200,240,3)*127/255;fov=true(200,240);name='constant';
 elseif k==2,I=cat(3,.2+.3*x/240,.25+.2*y/200,.3+.1*sin(x/11).*cos(y/9));fov=true(200,240);name='smooth_full';
 else,I=cat(3,.2+.3*x/240,.25+.2*y/200,.3+.1*sin(x/11).*cos(y/9));fov=((x-120).^2+(y-100).^2)<90^2;name='smooth_circle';
 end
 normalized=local_contrast_and_clahe(I,fov);stored=uint16(round(normalized*4096));
 cases(k)=struct('name',name,'input',I,'fov',fov,'normalized',normalized,'stored',stored);
 fprintf('%s finite=%d range=%g,%g\n',name,all(isfinite(normalized(:))),min(normalized(:)),max(normalized(:)));
end
save('synthetic_photo.mat','cases','-v7');
