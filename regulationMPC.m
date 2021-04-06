%% Init
clear; clc; close all;

addpath('./funcs/');

load('vars/BBP.mat', 'ss');
ss.UserData.x0 = [0.2 -0.1 0.3 -0.2 0 0 0 0]';
x0 = [0.2 -0.1 0.3 -0.2 0 0 0 0]';

%% Regulation MPC

ss = c2d(ss,.1); % discretization

cont.Q = 0.1*eye(size(ss.A,1));
cont.R = 0.2*eye(size(ss.B,2));

dim.N = 20; % prediciton horizon
dim.nx = size(ss.A,1);
dim.nu = size(ss.B,2);
dim.ny = size(ss.C,1);

[P,S]=predmodgen(ss,dim); % Generation of prediction model 

% Set some options for YALMIP and solver
options = sdpsettings('verbose',0,'solver','quadprog');

T=40;
x_0 = x0;
x(:,1) = x0;


for k=1:T
    % Write the cost function in quadratic form
    [H,h]=costgen(P(1:end-dim.nx,:),S(1:end-dim.nx,:),cont.Q,cont.R,dim,x_0); 

    % Solve the constrained optimization problem (with YALMIP)
    u_uncon = sdpvar(dim.nu*dim.N,1);                % define optimization variable

    Constraint=[];                  %define constraints

    Objective = 0.5*u_uncon'*H*u_uncon+h'*u_uncon;  %define cost function

    optimize(Constraint,Objective,options);  %solve the problem
    u_uncon=value(u_uncon);                  %assign the solution to uopt

    % Select the first input only
    u_rec(:,k) = u_uncon(1:dim.nu);

    % Compute the state/output evolution
    x(:,k+1)=ss.A*x_0 + ss.B*u_rec(:,k);

    % Update initial state for the next iteration
    x_0=x(:,k+1);

    clear u_uncon
end

figure(1),
t = linspace(0,T,T+1);  
%stairs(t, x(1,:), 'b', 'LineWidth', 1.3);

for i = 1:dim.nx
    hold on
    stairs(t, x(i,:), 'LineWidth', 1.3);
    hold off
end

legend('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8');

figure(2)
t = linspace(0,T-1,T);

for i = 1:dim.nu
    hold on
    stairs(t, u_rec(i,:), 'LineWidth', 1.3);
    hold off
end

legend('u1', 'u2', 'u3', 'u4');

clear u_rec x

x_0 = x0;
x(:,1) = x0;

%% Functions
function [T,S]=predmodgen(LTI,dim)
    %Prediction matrices generation
    %This function computes the prediction matrices to be used in the
    %optimization problem

    %Prediction matrix from initial state
    T=zeros(dim.nx*(dim.N+1),dim.nx);
    for k=0:dim.N
        T(k*dim.nx+1:(k+1)*dim.nx,:)=LTI.A^k;
    end

    %Prediction matrix from input
    S=zeros(dim.nx*(dim.N+1),dim.nu*(dim.N));
    for k=1:dim.N
        for i=0:k-1
            S(k*dim.nx+1:(k+1)*dim.nx,i*dim.nu+1:(i+1)*dim.nu)=LTI.A^(k-1-i)*LTI.B;
        end
    end

end

function [H,h]=costgen(T,S,Q,R,dim,x0)
    Qbar=kron(eye(dim.N),Q); 
    H=S'*Qbar*S+kron(eye(dim.N),R);   
    h=S'*Qbar*T*x0;
end
