clear;clc;

% World constants
%	g = 9.81;
%	mb = 0.05;
%	Ip = 0.12;
%	l = 0.3;
% 
% Functions:
%	a = (3/5)*g;
%	b = -(mb*g)/Ip;
%	c = l/(2*Ip);

load('BBP.mat', 'ss')

% Stability check;
%eig(A) % for stable MPC control all eig(A) < 1
   
% Controlability test:
%rank(A)
%Cont = ctrb(A,B);
%rank(Cont)

% define dimensions
nx = size(ss.A,1);
ny = size(ss.C,1);
nu = size(ss.B,2);
N = 5;