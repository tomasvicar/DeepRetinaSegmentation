close all;clear all;clc;
addpath('matlab_preprocessing')
path = 'G:\Sdílené disky\Retina GAČR\Data\databases\';
path_ubmi = 'G:\Sdílené disky\Retina GAČR\Měření na UBMI\Sada01\Sada_01\';

resolution = 35;  % pixels/degree - e.g. 25
output_folder = 'D:\DeepRetinaSegmentationData\data_preprocessed_dicom_35nnUNet';
task = 'Task535_Ophtalmo';

% mkdir(output_folder)

%%

load_avrdb_VA(resolution, path, [output_folder '\' task]);

load_iostar_VA(resolution, path, [output_folder '\' task]);

load_drive_VA(resolution, path, [output_folder '\' task]);

load_hrf_VA(resolution, path, [output_folder '\' task]);

load_drhagis_VA(resolution, path, [output_folder '\' task]);

load_ubmi_VA(resolution, path_ubmi, [output_folder '\' task]);