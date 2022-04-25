close all;clear all;clc;

path = 'G:\Sdílené disky\Retina GAČR\Data\databases\';

resolution = 12;  % pixels/degree - e.g. 25
output_folder = 'data_preprocessed_dicom_12';

mkdir(output_folder)
addpath('matlab_preprocessing')
%%

load_avrdb_VA(resolution, path, output_folder);

load_iostar_VA(resolution, path, output_folder);

load_drive_VA(resolution, path, output_folder);

load_hrf_VA(resolution, path, output_folder);

load_drhagis_VA(resolution, path, output_folder);

