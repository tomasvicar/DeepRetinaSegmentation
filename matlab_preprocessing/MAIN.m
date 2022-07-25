close all;clear all;clc;

path = 'C:\Users\vicar\Downloads\tmp_databases\';

%%  SET
resolution = 25;  % pixels/degree - e.g. 25
output_folder = 'D:\tmp_retina_hdf5_tmp';


% resolution = 12;  % pixels/degree - e.g. 25
% output_folder = 'D:\tmp_retina_hdf5_12';

% 
% %%
% 
% disp('load_aria')
% load_aria(resolution, path, output_folder);
% disp('load_avrdb')
% load_avrdb(resolution, path, output_folder);
% disp('load_chasedb1')
% load_chasedb1(resolution, path, output_folder);
% 
% % load_diaretdb0(resolution, path, output_folder);
% 
% % load_diaretdb1(resolution, path, output_folder);
% disp('load_drhagis')
% load_drhagis(resolution, path, output_folder);
% disp('load_drishtigs')
% load_drishtigs(resolution, path, output_folder);
% disp('load_drive')
% load_drive(resolution, path, output_folder);
% 
% % load_erlangen(resolution, path, output_folder);
% 
% % load_eophtha(resolution, path, output_folder);
% 
% % load_fire(resolution, path, output_folder);
% disp('load_g1020')
% load_g1020(resolution, path, output_folder);
% 
% % load_heimed(resolution, path, output_folder);
% disp('load_hrf')
% load_hrf(resolution, path, output_folder);
% 
% % load_idrid(resolution, path, output_folder);
% 
% % load_inspireavr(resolution, path, output_folder);
% disp('load_iostar')
% load_iostar(resolution, path, output_folder);
% 
% % load_messidor(resolution, path, output_folder);
% 
% % load_onhsd(resolution, path, output_folder);
% 
% % load_rcrgbma(resolution, path, output_folder);
% disp('load_refuge')
% load_refuge(resolution, path, output_folder);
% 
% % load_review(resolution, path, output_folder);
% 
% % load_ridb(resolution, path, output_folder);
% disp('load_riga')
% load_riga(resolution, path, output_folder);
% disp('load_rimone')
% load_rimone(resolution, path, output_folder);

% load_roc(resolution, path, output_folder);
disp('load_stare')
load_stare(resolution, path, output_folder);
disp('load_uoadr')
load_uoadr(resolution, path, output_folder);

% load_vicavr(resolution, path, output_folder);



