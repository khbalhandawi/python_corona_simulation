clearvars
clc
format compact
close all
format long
addpath support_functions

global index;
index = 0;

% x = [0.9, 0.0, 0.2];
x = [0.5, 0.5, 0.5];

f = Blackbox_call(x)


% output_filename = 'matlab_out_Blackbox.log';
% out_full_filename = ['data/',output_filename];
% fileID_out = fopen(out_full_filename,'r');
% f = textscan(fileID_out,'%f %f %f', 'Delimiter', ',');