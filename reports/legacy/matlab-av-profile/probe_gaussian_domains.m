load('synthetic_photo.mat','cases');I=cases(2).input;sigma=80/(55/25);
auto=imgaussfilt(I,sigma,'Padding','replicate');spatial=imgaussfilt(I,sigma,'Padding','replicate','FilterDomain','spatial');frequency=imgaussfilt(I,sigma,'Padding','replicate','FilterDomain','frequency');
corrected=(I-auto)./auto+.5;corrected=max(0,min(1,corrected));H=rgb2hsv(corrected);v_before=H(:,:,3);
save('synthetic_gaussian_domains.mat','I','auto','spatial','frequency','corrected','v_before','-v7');fprintf('auto_spatial=%g auto_frequency=%g\n',max(abs(auto(:)-spatial(:))),max(abs(auto(:)-frequency(:))));
