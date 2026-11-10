clear;Z=load('diaret_photometry_stages.mat','extrapolated','background');sigma=80/(55/25);
spatial=imgaussfilt(Z.extrapolated,sigma,'Padding','replicate','FilterDomain','spatial');frequency=imgaussfilt(Z.extrapolated,sigma,'Padding','replicate','FilterDomain','frequency');
spatial_max=max(abs(spatial(:)-Z.background(:)));frequency_max=max(abs(frequency(:)-Z.background(:)));spatial_unequal=nnz(spatial~=Z.background);frequency_unequal=nnz(frequency~=Z.background);
fprintf('auto_vs_spatial max=%.17g unequal=%d; auto_vs_frequency max=%.17g unequal=%d\n',spatial_max,spatial_unequal,frequency_max,frequency_unequal);
save('diaret_gaussian_domain.mat','spatial_max','frequency_max','spatial_unequal','frequency_unequal','-v7');
