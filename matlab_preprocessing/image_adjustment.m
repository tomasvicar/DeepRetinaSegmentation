function [I2,M1,M2,M3,F]=image_adjustment(im,rc,degree,M1,M2,M3, dat, fov)
    im=im2double(im);
    [rad,sloup,prumer,fov]=souradnice(im, dat, fov);
    
    minr=(rad-prumer)*(prumer<rad)+1*(prumer>=rad);
    maxr=(rad+prumer)*(size(im,1)>=(rad+prumer))+size(im,1)*(size(im,1)<(rad+prumer));
    minc=(sloup-prumer)*(prumer<sloup)+1*(prumer>=sloup);
    maxc=(sloup+prumer)*(size(im,2)>=(sloup+prumer))+size(im,2)*(size(im,2)<(sloup+prumer));

lengthI =  2*prumer;
% lengthI =  max(maxr-minr,maxc-minc);
I = im;    
I2=imresize(I,(rc*degree)/lengthI); 

F = fov;
F=imresize(F,(rc*degree)/lengthI); 

if length(M1)>5
    M1=imresize(M1,(rc*degree)/lengthI); 
end
if length(M2)>5
    M2=imresize(M2,(rc*degree)/lengthI);   
end
if length(M3)>5
    M3=imresize(M3,(rc*degree)/lengthI);  
end

    
    
    
    
    
% I = im(minr:maxr,minc:maxc,:);
% I2=imresize(I,(rc*degree)/length(I)); 
% 
% F = fov(minr:maxr,minc:maxc,:);
% F=imresize(F,(rc*degree)/length(I)); 
% 
% if length(M1)>5
%     M1 = M1(minr:maxr,minc:maxc,:);
%     M1=imresize(M1,(rc*degree)/length(I)); 
% end
% if length(M2)>5
%     M2 = M2(minr:maxr,minc:maxc,:);
%     M2=imresize(M2,(rc*degree)/length(I));   
% end
% if length(M3)>5
%     M3 = M3(minr:maxr,minc:maxc,:);
%     M3=imresize(M3,(rc*degree)/length(I));  
% end


end