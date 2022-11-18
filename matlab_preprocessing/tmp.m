clc;clear all;close all;

mm = imread("D:\DeepRetinaSegmentation\data_preprocessed\Images\aria_na_amd_10027.png");

shape = size(mm);



for k = 0:(300-1)
    k
    h5create('../../tmp.hdf5',['/aria_na_amd_10027' num2str(k)],shape,'Datatype','uint8','ChunkSize',[100 100 3],'Deflate',4)
%     ,'ChunkSize',shape,'Deflate',4
    
    h5write('../../tmp.hdf5',['/aria_na_amd_10027' num2str(k)],mm,[1,1,1],shape)
end

% with h5py.File('tmp.hdf5',"w") as f:
%     
%     start = time.time()
%     for k in range(3000):
%         if k%100 ==0:
%             print(k)
%             end = time.time()
%             print(end - start)
%             start = time.time()
%         dset_img = f.create_dataset('aria_na_amd_10027' + str(k), (shape[0],shape[1],3), dtype='u1')
%         dset_img[:,:,:] = mm