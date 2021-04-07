% Init
clear; clc; close all;
addpath('./funcs/');
load('vars/BBP.mat', 'ss');
ss = c2d(ss,.1); % discretization

%% Set constants

% tuning variables 
x0 = [0.1, 0, -0.05, 0.05, -0.04, 0, 0, 0]';
cont.Q = 0.1*eye(size(ss.A,1));
cont.R = 0.2*eye(size(ss.B,2));
dim.N = 15; % prediciton horizon
d = [0.1; 1]; % disturbance
y_ref = [0.8780; 0.6590]; % reference

% dimnsions
dim.nx = size(ss.A,1);
dim.nu = size(ss.B,2);
dim.ny = size(ss.C,1);
dim.nd = size(d,1);

%% Set augmented dynamics
Cd = diag([0.5 2]);
Bd = [1, 0.5; 0.4, 0; 0, 1; 0.3, 0; 0, 0.5; 1, 0; 0, 1; 0, 0.1]; % should have 8 rows        

augss.A = [ss.A, Bd; zeros(dim.nd,dim.nx), eye(dim.nd)];
augss.B = [ss.B; zeros(dim.nd,dim.nu)];
augss.C = [ss.C Cd];

% Check stabality - works
% aug_sys = [eye(dim.nx) - ss.A, -Bd; ss.C, Cd];
% rank(aug_sys) % rank <= nx + ny   

Pdare = blkdiag(idare(ss.A, ss.B, cont.Q, cont.R), zeros(dim.nd));
cont.Q = blkdiag(cont.Q, zeros(dim.nd)); 

dim.nx = dim.nx + dim.nd; 
x0 = [x0; d];

%% simulation time
time = 6;
dt = 0.1; % timestep
T = linspace(0, time, time/dt);
T_1 = [T(1,:),time+dt];
t = length(T);

%% Output feedback MPC

[P,S] = predmodgen(augss,dim); % Generation of prediction model 
[H,h] = costgen(Pdare,P,S,cont,dim); 

% Receding horizon implementation
x = zeros(dim.nx, t+1);
y = zeros(dim.ny, t+1);
u_rec = zeros(dim.nu, t);
x_hat = zeros(dim.nx, t+1);

x(:,1) = x0;
x_hat(:,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
y(:,1) = augss.C*x0;

L = place(augss.A', augss.C', [0.5; 0.4; 0.45; 0.6; 0.65; 0.5; 0.54; 0.3; 0.43; 0.45])'; %observer gain

for k=1:t
    x_0 = x(:,k);  
    dhat = x_hat(end-dim.nd+1:end,k);
    
    %Compute optimal ss (online, at every iteration)
    A = [eye(dim.nx-dim.nd)-ss.A, -ss.B; ss.C, zeros(dim.ny,dim.nu)];
    b = [Bd*dhat; y_ref-Cd*dhat];
    
    options1 = optimoptions(@quadprog); 
    options1.OptimalityTolerance=1e-20;
    options1.ConstraintTolerance=1.0000e-15;
    options1.Display='off';
    
    O = blkdiag(zeros(dim.nx-dim.nd),eye(dim.nu));
    o = zeros(dim.nx-dim.nd+dim.nu,1);
    xur = quadprog(O,o,[],[],A,b,[],[],[],options1);
    xr = xur(1:dim.nx);
    ur = xur(dim.nx+1:end);

    xre = [xr; dhat];
    
    uostar = sdpvar(dim.nu*dim.N,1);                                 %define optimization variable
    Constraint=[];                                                     %define constraints
    Objective = 0.5*uostar'*H*uostar+(h*[x_0; xre; ur])'*uostar;    %define cost function
    optimize(Constraint,Objective);                                    %solve the problem
    uostar=value(uostar);      

    % Select the first input only
    u_rec(:,k)=uostar(1:dim.nu);

    % Compute the state/output evolution
    x(:,k+1) = augss.A*x_0 + augss.B*u_rec(:,k);
    y(:,k+1) = augss.C*x(:,k+1);
    
    clear u_uncon
        
    % Update state estimation using Luenberg
    x_hat(:,k+1) = augss.A*x_hat(:,k) + augss.B*u_rec(:,k) + L*(y(:,k)-augss.C*x_hat(:,k));
end

function [H,h] = costgen(Pdare,P,S,cont,dim)
    Qbar = blkdiag(kron(eye(dim.N),cont.Q),Pdare);
    Rbar = kron(eye(dim.N),cont.R);
    H = S'*Qbar*S+Rbar;   
    hx0 = S'*Qbar*P;
    hxref = -S'*Qbar*kron(ones(dim.N+1,1),eye(dim.nx));
    huref = -Rbar*kron(ones(dim.N,1),eye(dim.nu));
    h = [hx0 hxref huref];
end

function [P,S] = predmodgen(ss,dim)
    % Prediction matrix from initial state
    P = zeros(dim.nx*(dim.N+1),dim.nx);
    for k = 0:dim.N
        P(k*dim.nx+1:(k+1)*dim.nx,:) = ss.A^k;
    end

    % Prediction matrix from input
    S = zeros(dim.nx*(dim.N+1),dim.nu*(dim.N));
    for k  =1:dim.N
        for i = 0:k-1
            S(k*dim.nx+1:(k+1)*dim.nx,i*dim.nu+1:(i+1)*dim.nu) = ss.A^(k-1-i)*ss.B;
        end
    end
end
