function [rad,sloup,prumer,fov]=souradnice(im, database,fov)
database=string(database);

if length(fov)>5
    final_closed=fov;
else
    problem=0;
    im=im2double(rgb2gray(im));
    %% THRESHHOLD 1 
    if database=='diaretdb' || database =='review'
    prah1=min(multithresh(im,4))*0.1;
    elseif database=='drishtigs' || database=='idrid' || database=='messidor' || database=='riga' || database=='roc' || database=='vicavr' || database=='avrdb'
    prah1=min(multithresh(im,5));
    elseif database=='stare'
    prah1=min(multithresh(im,3));  
    elseif database=='eophtha' || database=='ridb'
    prah1=min(multithresh(im,5))*0.5;
    else
    prah1=0;
    end
    %% THRESHOLD 2
    if database=='inspireavr'
        prah2=max(im(:))*0.4;
    else
    prah2=max(im(:))*0.8;
    end
    
    highmask = im>prah2;
    lowmask = bwlabel((im>prah1));
    final = ismember(lowmask,unique(lowmask(highmask)));
    se = strel('disk',10);
    final_closed= imclose(final,se);
% 
%     figure(1)
%     subplot(321)
%     imshow(im,[])
%     subplot(322)
%     hist(im(:),250)
%     hold on
%     stem([prah1 prah2],[1000,1000],'r','filled');hold off
%     subplot(323)
%     imshow(logical(lowmask),[])
%     subplot(324)
%     imshow(highmask,[])
%     subplot(325)
%     imshow(final,[])
    
    bpx=1-((sum(final_closed(:)))/(size(im,1)*size(im,2))); % number of blasck pixels

    %% kdyz to nefunguje
    prah=0.1;
    if database=='aria' || database=='g1020'
        bpx=0.1343; % g1020 .. bpx=0.1272; idrid .. bpx=0.3105; refuge .. bpx=0.247; review .. bpx=0.226; 
    elseif database=='diaretdb'
        bpx=0.1634;
    elseif database=='stare'
        bpx=0.269;
    elseif database=='drishtigs'
        bpx=0.1568;
        prah=0.05;
        if size(im,2)>2200
           bpx=0.2929;
        end
    elseif database=='eophtha'
        bpx=0.45;
        prah=0.15;
    end   
        
    wpx=1-bpx;
  
    if ((sum(final_closed(:)))/(size(im,1)*size(im,2)))> (wpx+prah) || ((sum(final_closed(:)))/(size(im,1)*size(im,2)))< (wpx-prah)
        [counts,edges] = histcounts(im(:),250);
        cc = cumsum(counts);
        prah1=edges(min(find(cc>bpx*(size(im,1)*size(im,2)))));
        prah2=max(im(:))*0.8;
        highmask = im>prah2;
        lowmask = bwlabel((im>prah1));
        final = ismember(lowmask,unique(lowmask(highmask)));
        se = strel('disk',10);
        final_closed= imclose(final,se);
        hr=edge(final_closed);
        problem=1;
%         problem=((sum(final_closed(:)))/(size(im,1)*size(im,2)))> (wpx+prah) || ((sum(final_closed(:)))/(size(im,1)*size(im,2)))< (wpx-prah);
    end
% 
%     if problem
%         figure(2)
%         subplot(321)
%         imshow(im,[])
%         subplot(322)
%         hist(im(:),250)
%         hold on
%         stem([prah1 prah2],[1000,1000],'r','filled');hold off
%         subplot(323)
%         imshow(logical(lowmask),[])
%         subplot(324)
%         imshow(highmask,[])
%         subplot(325)
%         imshow(final_closed,[])
%         a=5;
%     end

fov=imfill(final_closed,'hole');
end
%%
if database=='rimone'
    hr=fov;
    fov=ones(size(fov));
elseif database=='erlangen' || database=='erlangen_videodata'
    hr=ones(size(fov));
    hr(:,2:end-1)=0;
elseif database=='roc' || database=='riga'
hr=zeros(size(final_closed));
hr(20:end-20,20:end-20)=edge(final_closed(20:end-20,20:end-20));
else
hr=edge(final_closed);
end

for i=1:size(hr,1)
    ind=find(hr(i,:));
    if length(ind)>=2
    sloupec(i)=(max(ind)-min(ind))/2 + min(ind);
    end
end
for i=1:size(hr,2)
    ind=find(hr(:,i));
    if length(ind)>=2
    radek(i)=(max(ind)-min(ind))/2 + min(ind);
    end
end
sloup=round(mode(sloupec(logical(sloupec))));
rad=round(mode(radek(logical(radek))));
X=meshgrid(-sloup+1:1:size(im,2)-sloup,1:size(im,1));
Y=meshgrid(-rad+1:1:size(im,1)-rad,1:size(im,2))';
Z=sqrt(X.^2 + Y.^2);
prumer=round(mode(Z(logical(hr))))+1;
end
    

