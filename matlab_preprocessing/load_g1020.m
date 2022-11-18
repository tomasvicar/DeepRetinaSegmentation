function[] = load_g1020(rc, path, out_f)
degree = 45;

%% G1020
images = dir([path 'G1020\*.jpg']);
for i=1:length(images)
    
    in=images(i).name(1:end-4);
    
    im=imread([path 'G1020\' images(i).name ]);
    %% kuchani disku a cupu z .json
    fid = fopen([path 'G1020\' in '.json' ]); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    val = jsondecode(str);
    for j=1:length(val.shapes)
        dd(j)=string(val.shapes(j).label);
        if length(unique(dd))~= length(dd)
            [C,ia,ic] = unique(dd);
            kde=find(ic==mode(ic));
            if length(val.shapes(kde(1)).points)>length(val.shapes(kde(2)).points); dd(kde(1))=('disc'); dd(kde(2))=('cup');end
            if length(val.shapes(kde(1)).points)<length(val.shapes(kde(2)).points); dd(kde(1))=('cup'); dd(kde(2))=('disc');end

        end
    end
    [m, o, ~]=size(im);
    if sum(dd=='disc')
    disc_coor=val.shapes(find(dd=='disc')).points';
    disc = logical(sum(insertShape(zeros(m,o),'FilledPolygon',disc_coor(:)',...
    'Color', 'white','Opacity',1),3));
    else
        disc=0;
    end
    if sum(dd=='cup')
    cup_coor=val.shapes(find(dd=='cup')).points';
    cup = logical(sum(insertShape(zeros(m,o),'FilledPolygon',cup_coor(:)',...
    'Color', 'white','Opacity',1),3));
    else
        cup=0;
    end
    clear dd
    %%

    [I,D,C,~,fov]=image_adjustment(im,rc,degree,disc,cup,0, 'g1020', 0);
    
    ind=strfind(in,'_');
    in(ind)=[];
    imname= [ 'g1020_na_na_'  in(6:end)  ];


    imwrite_single(I,[out_f '\Images\' imname '.tiff'])
    imwrite_single(fov,[out_f '\Fov\' imname '_fov.tiff'])
    if length(D)>1
    imwrite_single(D,[out_f '\Disc\' imname '_disc.tiff'])
    end
    if length(C)>1
    imwrite_single(C,[out_f '\Cup\' imname '_cup.tiff'])
    end
end

end