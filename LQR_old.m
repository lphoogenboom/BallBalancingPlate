%% Init
clear;
clc;
load('BBP.mat', 'ss')
ss = c2d(ss,.1); % discretization

%% LQR

% tuning parameters
x0 = [0.2 -0.1 0.3 -0.2 0 0 0 0]';
cont.Q = 1*eye(size(ss.A,1));
cont.R = 2*eye(size(ss.B,2));

[P,L,G] = dare(ss.A,ss.B,cont.Q,cont.R); 

dim.nx = size(ss.A,1);
dim.nu = size(ss.B,2);
dim.ny = size(ss.C,1);

% simulation
time = 6;
dt = 0.1; % timestep
T = linspace(0, time, time/dt);
T_1 = [T(1,:),time+dt];
t = length(T);

[x,u] = lqr(x0,t,ss,dim,G);

%% Visuals
figure(1)
hold on
stairs(T_1, x(1,:), 'b', 'LineWidth', 1.3);
stairs(T_1, x(3,:), 'r', 'LineWidth', 1.3);
hold off

legend('xb', 'yb');
%% Functions

function [x,u] = lqr(x0,t,ss,dim,G)
    x = zeros(dim.nx,length(t));   
    u = zeros(dim.nu,length(t));
    x(:,1) = x0;

    for k = 1:t
        u(:,k) = -G*x(:,k);
        x(:,k+1) = ss.A*x(:,k) + ss.B*u(:,k);
    end
end