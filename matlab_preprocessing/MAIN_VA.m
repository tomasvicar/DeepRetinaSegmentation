close all;clear all;clc;
addpath('matlab_preprocessing')
path = 'G:\Sdílené disky\Retina GAČR\Data\databases\';
path_ubmi = 'G:\Sdílené disky\Retina GAČR\Měření na UBMI\Sada01\Sada_01\';

resolution = 25;  % pixels/degree - e.g. 25
output_folder = 'data_preprocessed_dicom_25N';

mkdir(output_folder)

%%
% 
% load_avrdb_VA(resolution, path, output_folder);
% 
% load_iostar_VA(resolution, path, output_folder);
% 
% load_drive_VA(resolution, path, output_folder);
% 
% load_hrf_VA(resolution, path, output_folder);
% 
% load_drhagis_VA(resolution, path, output_folder);

load_ubmi_VA(resolution, path_ubmi, output_folder);