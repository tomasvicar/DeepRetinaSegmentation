function[] = load_riga(rc, path, out_f)

%% RIGA
degree = 45;
images=[];
images = dir([path 'RIGA\MESSIDOR\*prime.tif']);
images = [images; dir([path 'RIGA\BinRushed\BinRushed1-Corrected\*prime.jpg'])];
images = [images; dir([path 'RIGA\BinRushed\BinRushed2\*prime.jpg'])];
images = [images; dir([path 'RIGA\BinRushed\BinRushed3\*prime.jpg'])];
images = [images; dir([path 'RIGA\BinRushed\BinRushed4\*prime.jpg'])];
images = [images; dir([path 'RIGA\Magrabia\Magrabia\MagrabiaMale\*prime.tif'])];
images = [images; dir([path 'RIGA\Magrabia\Magrabia\MagrabiFemale\*prime.tif'])];

se = strel('diamond',3);
for i=1:length(images)
    
    in=images(i).name(1:end-9);
    
    im=imread([images(i).folder '\' images(i).name ]);
    
    fold=images(i).folder;
    ind=strfind(fold,'\');
    fol=fold(ind(end)+1:end);
    fol(strfind(fol,'-'))=[];
    imname= [ 'riga_na_na_'  fol in  ];
    if fold(ind(end)-8:ind(end)-1)==string('Magrabia')
        degree=35;
    end
    
    for h=1:6
        if fol(1:5)==string('BinRu') && ~((fol(end-5:end)==string('rected') && h==3)&& str2num(in(6:end))<25 )
        gt=imread([images(i).folder '\' in '-' num2str(h) '.jpg' ]);
        else
        gt=imread([images(i).folder '\' in '-' num2str(h) '.tif' ]);
        end
        mask1=(rgb2gray(im2double(im))-rgb2gray(im2double(gt)));
        mask2=mat2gray(mask1);
        [mask3,~] = imgradient(mask2);
        mask4=imbinarize(mask3,0.4);
        mask=bwlabel(imclose(imclose(mask4,se),se));
        S = regionprops(mask,'area');
        [B,Ind] = sort(struct2array(S));
        disc=zeros(size(mask));
        disc(mask==Ind(end))=1;
        disc=imfill(disc,'hole');
        if length(Ind)<2
            ed=edge(disc);
            ed_disc = imdilate(ed,strel('diamond',12));
            mask(ed_disc)=mask(ed_disc)./2;
            mask=mask*2;
            S = regionprops(mask,'area');
            [B,Ind] = sort(struct2array(S));
        end
            
        cup1=zeros(size(mask));
        cup1(mask==Ind(end-1))=1;
        cup=imfill(cup1,'hole');
        
       if ((sum(disc(:)==cup(:)))/size(disc,1)*size(disc,2))>=0.99
            ed=edge(disc);
            ed_disc = imdilate(ed,strel('diamond',12));
            mask(ed_disc)=mask(ed_disc)./2;
            mask=mask*2;
            S = regionprops(mask,'area');
            [B,Ind] = sort(struct2array(S));
            cup=zeros(size(mask));
            cup(mask==Ind(end-1))=1;
            cup=imfill(cup,'hole');
        end
  
        [I,D,C,~, fov]=image_adjustment(im,rc,degree,disc,cup,0, 'riga', 0); 

        if h==1
        imwrite_single(D,[out_f '\Disc\' imname '_disc.tiff'])
        imwrite_single(C,[out_f '\Cup\' imname '_cup.tiff'])
        else
        imwrite_single(D,[out_f '\Disc\' imname '_disc' num2str(h) '.tiff'])
        imwrite_single(C,[out_f '\Cup\' imname '_cup' num2str(h) '.tiff'])
        end       
    end
    
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])
    imwrite_single(I,[out_f '\Images\' imname '.tiff'])

end
end