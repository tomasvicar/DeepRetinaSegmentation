function []=imwrite_single2(data,name)


output_folder = '../../data_preprocessed_hdf5';%%%%%%zmenit v imwrite_single

name = replace(name,output_folder,'');
name = replace(name,'\','/');
name = replace(name,'.tiff','');


if contains(name,'_fov')||contains(name,'_ves')||contains(name,'_disc')||contains(name,'_cup')
    
    if length(size(data))==3
        data = data(:,:,1);
    end
    
    T = mean([max(data(:)) ,min(data(:))]);
    if (T - min(data(:)))==0
       T =  Inf;
    end
    

    
    data = data>T;
    
    data = uint8(data*255);
    
    shape = size(data);
    xxx = [output_folder '/dataset_pretrain.hdf5'];
    h5create(xxx,name,shape,'Datatype','uint8','ChunkSize',[100 100],'Deflate',4)

    h5write(xxx,name,data,[1,1],shape)
    
    return
end


% imwrite(,replace(name,'.tiff','.png'))

data = uint8(data*255);

shape = size(data);
xxx = [output_folder '/dataset_pretrain.hdf5'];
h5create(xxx,name,shape,'Datatype','uint8','ChunkSize',[100 100 3],'Deflate',4)

h5write(xxx,name,data,[1,1 1],shape)


    
    
    
    
end



