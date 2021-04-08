%% Init
clear; 
clc; 
close all;
clear functions;
% addpath('./funcs/');
load('vars/BBP.mat', 'ss');
ss = c2d(ss,.1); % discretization

%% Regulation MPC

% Tuning variables
cont.Q = .5*eye(size(ss.A,1));
cont.R = .3*eye(size(ss.B,2));
% x0 = [.01, 0.01, .03, -.05, 0, 0, 0, 0]'; inside temrinal set
x0 = [-.1, -0.05, .12, -.1, .01, .01, .01, .01]';
dim.N = 15; % prediciton horizon

dim.nx = size(ss.A,1);
dim.nu = size(ss.B,2);
dim.ny = size(ss.C,1);

[P,S]=predmodgen(ss,dim); % Generation of prediction model 
Pdare = idare(ss.A, ss.B, cont.Q, cont.R);

% Set some options for YALMIP and solver
options = sdpsettings('verbose',0,'solver','quadprog');

% simulation
time = 20;
dt = 0.2; % timestep
T = linspace(0, time, time/dt);
T_1 = [T(1,:),time+dt];
t = length(T);

% initial conditions
x_0 = x0;
x(:,1) = x0;
x_rec = [];

for k=1:t
    fprintf('Sim time: '+string(k*dt)+'/'+string(t*dt)+'\n')
    % Write the cost function in quadratic form
    [H,h]=costgen(P(1:end-dim.nx,:),S(1:end-dim.nx,:),cont.Q,cont.R,dim,x_0);

    % Solve the constrained optimization problem (with YALMIP)
    u_con = sdpvar(dim.nu*dim.N,1);   % define optimization variable
	x_con = sdpvar(length(x(:,1)),1);

    Constraint = [abs(u_con)<=2.5,... 
                  abs(x_con(1))<=.15, abs(x_con(2))<=2, abs(x_con(3))<=.15,...
                  abs(x_con(4))<=2, abs(x_con(5))<=pi/4,abs(x_con(6))<=3,...
                  abs(x_con(7))<=pi/4, abs(x_con(8))<=3]; %define constraints
              
    if k==t
        Constraint = [x_0'*Pdare(1:8,1:8)*x_0<=0.56 ,abs(u_con)<=2.5,... 
                        abs(x_con(1))<=.15, abs(x_con(2))<=2, abs(x_con(3))<=.15,...
                        abs(x_con(4))<=2, abs(x_con(5))<=pi/4,abs(x_con(6))<=3,...
                        abs(x_con(7))<=pi/4, abs(x_con(8))<=3]; %define constraints
    end

    Objective = 0.5*u_con'*H*u_con+h'*u_con;  %define cost function
	
    optimize(Constraint,Objective,options); %solve the problem
    
	u_con=value(u_con); %assign the solution to uopt
	x_con = value(x_con);
	
    % Select the first input only
    u_rec(:,k) = u_con(1:dim.nu);
	x_rec = [x_rec x_con]; % store state history for insight

    % Compute the state/output evolution
    x(:,k+1)=ss.A*x_0 + ss.B*u_rec(:,k);

    % Update initial state for the next iteration
    x_0=x(:,k+1);

    clear u_con
end

%%

figure(1)
clf; hold on; grid on;
stairs(T_1, x(1,:), 'b', 'LineWidth', 1.3);
stairs(T_1, x(3,:), 'r', 'LineWidth', 1.3);

figure(2)
clf; hold on; grid on;
for i = 1:dim.nu
    stairs(T, u_rec(i,:), 'LineWidth', 1.3);
end


figure(3)
clf; hold on; grid on;
stairs(T_1, x(2,:), 'b', 'LineWidth', 1.3);
stairs(T_1, x(4,:), 'r', 'LineWidth', 1.3);

figure(4)
clf; hold on; grid on;
stairs(T_1, x(5,:), 'b', 'LineWidth', 1.3);
stairs(T_1, x(7,:), 'r', 'LineWidth', 1.3);

figure(5)
clf; hold on; grid on;
stairs(T_1, x(6,:), 'b', 'LineWidth', 1.3);
stairs(T_1, x(8,:), 'r', 'LineWidth', 1.3);


%% Functions
function [T,S]=predmodgen(ss,dim)
    % Prediction matrix from initial state
    T = zeros(dim.nx*(dim.N+1),dim.nx);
    for k = 0:dim.N
        T(k*dim.nx+1:(k+1)*dim.nx,:)=ss.A^k;
    end

    % Prediction matrix from input
    S = zeros(dim.nx*(dim.N+1),dim.nu*(dim.N));
    for k = 1:dim.N
        for i = 0:k-1
            S(k*dim.nx+1:(k+1)*dim.nx,i*dim.nu+1:(i+1)*dim.nu)=ss.A^(k-1-i)*ss.B;
        end
	end
end

function [H,h]=costgen(T,S,Q,R,dim,x0)
    Qbar=kron(eye(dim.N),Q); 
    H=S'*Qbar*S+kron(eye(dim.N),R);   
    h=S'*Qbar*T*x0;
end