%% Init
clear; clc; clf;

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

function [H,h]=costgen(P,S,Q,R,dim)
    Qbar=blkdiag(kron(eye(dim.N),Q),zeros(dim.nx)); 
    H=S'*Qbar*S+kron(eye(dim.N),R);   
    h=S'*Qbar*P;
end