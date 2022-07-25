function []=imwrite_2_h5(data,output_folder,name)



name = replace(name,'\','/');
output_folder = replace(output_folder,'\','/');

if contains(name,'_fov')||contains(name,'_ves')||contains(name,'_disc')||contains(name,'_cup')
    
    if length(size(data))==3
        data = data(:,:,1);
    end
    
    T = mean([max(data(:)) ,min(data(:))]);
    if (T - min(data(:)))==0
       T =  Inf;
    end
    

    data = data>T;
    
    ChunkSize = [100 100];
    start = [1 1];
else
    ChunkSize = [100 100 3];
    start = [1 1 1];
end



data = uint8(data*255);

shape = size(data);
tmp = split(output_folder,'/');
xxx = [output_folder '/' tmp{end} '.hdf5'];
% try
h5create(xxx,name,shape,'Datatype','uint8','ChunkSize',ChunkSize,'Deflate',4)
% end
h5write(xxx,name,data,start,shape)


    
    
end




