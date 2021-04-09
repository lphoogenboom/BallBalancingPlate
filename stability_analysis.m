%% Init
clear; clc; close all;
addpath('./funcs/');
load('vars/BBP.mat', 'ss');
x0 = [0.12, 0, -0.08, 0.1, 0, 0, 0, 0]';

%% Model definition

ss = c2d(ss,.1); % discretization

cont.Q = 0.5*eye(size(ss.A,1));
cont.R = 0.3*eye(size(ss.B,2));

dim.N = 15; % prediciton horizon
dim.nx = size(ss.A,1);
dim.nu = size(ss.B,2);
dim.ny = size(ss.C,1);

%% Model stability

eig(ss.A); % for stable MPC control all eig(A) < 1
   
% Controlability test:
A_rank = rank(ss.A);
Cont = ctrb(ss.A,ss.B);
Cont_rank = rank(Cont);

%% Solve DARE

[Pdare,K,L] = idare(ss.A,ss.B,cont.Q,cont.R); 

[V,D] = eig(Pdare);
M = max(abs(V), [], 'all');
c = 0.5/M; % 0.5

%% Regulation MPC

[P,S]=predmodgen(ss,dim); % Generation of prediction model 

% Set some options for YALMIP and solver
options = sdpsettings('verbose',0,'solver','quadprog');

% simulation
time = 6;
dt = 0.1; % timestep
T = linspace(0, time, time/dt);
T_1 = [T(1,:),time+dt];
t = length(T);

% initial conditions
x_0 = x0;
x(:,1) = x0;

% cost vunction values
l_ux = zeros(1,length(T));
Vf = zeros(1,length(T));
VfP = zeros(1,length(T));

for k=1:t
    % Write the cost function in quadratic form
    [H,h]=costgen(P(1:end-dim.nx,:),S(1:end-dim.nx,:),cont.Q,cont.R,dim,x_0);

    % Solve the constrained optimization problem (with YALMIP)
    u_uncon = sdpvar(dim.nu*dim.N,1);   % define optimization variable
	x_con = sdpvar(length(x(:,1)),8);

    Constraint = [abs(u_uncon)<=1.5, abs(x_con(1))<=.15, abs(x_con(2))<=2, abs(x_con(3))<=.15,...
                  abs(x_con(4))<=2, abs(x_con(5))<=pi/4,abs(x_con(6))<=3,abs(x_con(7))<=pi/4,...
                  abs(x_con(8))<=3]; %define constraints

    Objective = 0.5*u_uncon'*H*u_uncon + h'*u_uncon;  %define cost function

    optimize(Constraint,Objective,options); %solve the problem
    u_uncon=value(u_uncon); %assign the solution to uopt

    % Select the first input only
    u_rec(:,k) = u_uncon(1:dim.nu);

    % find costs at x(k), u(k)
    l_ux(:,k) = 0.5*x_0'*cont.Q*x_0 + 0.1*u_rec(:,k)'*cont.R*u_rec(:,k);
    Vf(:,k) = 0.5*x_0'*Pdare*x_0;
    
    % Compute the state/output evolution
    x(:,k+1) = ss.A*x_0 + ss.B*u_rec(:,k);
    
    % Update initial state for the next iteration
    x_0=x(:,k+1);
    
    % find cost at x(k+1)
    VfP(:,k) = 0.5*x_0'*Pdare*x_0;

    clear u_uncon
end

%% plot costs over time
figure(1),
stairs(T, VfP-Vf, 'b', 'LineWidth', 1.3);
hold on
stairs(T, -l_ux, 'r', 'LineWidth', 1.3);
hold off

legend('Vf(x(k+1))-Vf(x(k))', '-l(x(k),u(k))');

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















