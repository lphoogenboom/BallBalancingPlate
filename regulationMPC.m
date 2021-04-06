%% Init
clear; clc; close all;

addpath('./funcs/');

load('vars/BBP.mat', 'ss');
ss.UserData.x0 = [0.2 -0.1 0.3 -0.2 0 0 0 0]';
x0 = [0.2 -0.1 0.3 -0.2 0 0 0 0]';
%% Regulation MPC

ss = c2d(ss,.1); % discretization

cont.Q = 1*eye(size(ss.A,1));
cont.R = 2*eye(size(ss.B,2));

dim.N = 20; % prediciton horizon
dim.nx = size(ss.A,1);
dim.nu = size(ss.B,2);
dim.ny = size(ss.C,1);

[P,S]=predmodgen(ss,dim);            %Generation of prediction model 
[H,h]=costgen(P,S,cont.Q,cont.R,dim);  %Writing cost function in quadratic form