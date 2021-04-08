%% Init
clear;
clc;
clear functions;
load('./vars/BBP.mat', 'ss')
ss = c2d(ss,.1); % discretization

%% LQR

% tuning parameters
x0 = [.1, 0.1, .1, -.15, 0, 0, 0, 0]';
cont.Q = .5*eye(size(ss.A,1));
cont.R = .3*eye(size(ss.B,2));

[P,L,G] = dare(ss.A,ss.B,cont.Q,cont.R); 

dim.nx = size(ss.A,1);
dim.nu = size(ss.B,2);
dim.ny = size(ss.C,1);

% simulation
time = 20;
dt = 0.2; % timestep
T = linspace(0, time, time/dt);
T_1 = [T(1,:),time+dt];
t = length(T);

[x,u] = lqr(x0,t,ss,dim,G);

%% Visuals
figure(1)
hold on; grid on;
stairs(T_1, x(1,:), 'LineWidth', 1.3);
stairs(T_1, x(3,:), 'LineWidth', 1.3);
legend('MPC xb', 'MPC xy','LQR xb', 'LQR yb');

figure(2)
hold on; grid on;
plot(T,u(1,:))
plot(T,u(2,:))
plot(T,u(3,:))
plot(T,u(4,:))
legend('MPCu1', 'MPCu2', 'MPCu3', 'MPCu4','LQRu1', 'LQRu2', 'LQRu3', 'LQRu4');

figure(3)
hold on; grid on;
stairs(T_1, x(2,:), 'LineWidth', 1.3);
stairs(T_1, x(4,:), 'LineWidth', 1.3);
legend('MPC xb', 'MPC xy','LQR xb', 'LQR yb');

figure(4)
hold on; grid on;
stairs(T_1, x(5,:), 'LineWidth', 1.3);
stairs(T_1, x(7,:), 'LineWidth', 1.3);
legend('MPC xb', 'MPC xy','LQR xb', 'LQR yb');

figure(5)
hold on; grid on;
stairs(T_1, x(6,:), 'LineWidth', 1.3);
stairs(T_1, x(8,:), 'LineWidth', 1.3);
legend('MPC xb', 'MPC xy','LQR xb', 'LQR yb');

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