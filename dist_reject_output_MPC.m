clear; clc; close all; clear functions;
addpath('./funcs/');
load('vars/BBP.mat', 'ss');
ss = c2d(ss,.1); % discretization
clear functions;

%% State space
A = ss.A;
B = ss.B;
C = ss.C;
Cd = 1*eye(2);
Bd = 0.1*[0.3, 0.5; 0.3, 0.5; 0.5, 1; 1, 0.5; 0.4, 0.36; 0.54, 0.59; 0.56, 0.48; 0.38, 0.62];

% define dimensions
nx = size(A,1);
ny = size(C,1);
nu = size(B,2);
nd = 2;
N = 20;

% tuning variables
Q_gain = 0.3; 
R_gain = 0.11;
x0 = [0,0,0,0,0,0,0,0]';

Q = Q_gain*eye(nx);   
R = R_gain*eye(nu);

% observer gain
L1 = [0.1, 0.05; 0.05, 0.1];          
[P,~,~] = dare(A,B,Q,R);


% augmented state space
Ae = [A, Bd; zeros(nd, nx), eye(nd)];
Be = [B; zeros(nd,nu)];
Ce = [C, Cd];

Qe = blkdiag(Q, zeros(nd));           
Re = R;                                  
Pe = blkdiag(P, zeros(nd)); 
nxe = nx + nd;

[Te,Se] = predict_model(Ae,Be,nxe,nu,N);  
[He,he] = cost_model(Se,Te,nxe,nu,N,Qe,Pe,Re); 


%% simulation time
time = 30;
dt = 0.2; % timestep
T = linspace(0, time, time/dt);
T_1 = [T(1,:),time+dt];
t = length(T);

%% disturbance settings
d = zeros(2, t);

for i = 1:t
    if i >= 20 && i <= 30
       d(1,i) = 0.007;
       d(2,i) = -0.009;
    elseif i >= 100
       d(1,i) = -0.009;
       d(2,i) = 0.007;
    end
end

% some reference trajactory
y_ref = zeros(2, t);
% f = 1.5;

for i = 1:t
   if i >= 70
       y_ref(1,i) = 0.05;
       y_ref(2,i) = -0.05;
   end
end
      
% derive optimal inputs
x = zeros(nx,t+1);
x(:,1) = x0;
u_rec = zeros(nu,t);
y = zeros(ny,t);

dhat = zeros(nd,t+1);
xhat = zeros(nx,t+1);
dhat(:,1) = [0;0];
xhat(:,1) = [0;0;0;0;0;0;0;0];

[~,~, Lgain] = idare(Ae', Ce', Qe, R_gain*eye(2));
L = place(Ae', Ce', Lgain)'; %observer gain
Lx = L(1:nx,:);
Ld = L(nx+1:end,:);

for k=1:t
    y(:,k) = C*x(:,k) + Cd*d(:,k);
    
    Ac = [eye(nx)-A,-B; 
          C,zeros(ny,nu)];
    bc = [Bd*dhat(:,k); 
          y_ref(:,k)-Cd*dhat(:,k)];

    H0 = blkdiag(zeros(nx),eye(nu));
    h0 = zeros(nx+nu,1);
    xur = quadprog(H0,h0,[],[],Ac,bc,[],[],[]);
    xr = xur(1:nx);
    ur = xur(nx+1:end);

    u_con = sdpvar(nu*N,1);   % define optimization variable
    x_con = sdpvar(length(x(:,1)),1);

    Constraint=[abs(u_con)<=1.0,... 
                abs(x_con(1))<=.15, abs(x_con(2))<=2, abs(x_con(3))<=.15,...
                abs(x_con(4))<=2, abs(x_con(5))<=pi/4,abs(x_con(6))<=3,...
                abs(x_con(7))<=pi/4, abs(x_con(8))<=3];  % add contraints later

    %Constraint=[];
    Objective = 0.5*u_con'*He*u_con+(he*[x(:,k); d(:,k); xr; dhat(:,k); ur])'*u_con;
    optimize(Constraint,Objective);
    u_con = value(u_con);


    % Select the first input only
    u_rec(:,k) = u_con(1:nu);

    % Compute the state/output evolution
    x(:,k+1) = A*x(:,k) + B*u_rec(:,k) + Bd*d(:,k);

    clear u_con

    % Update disturbance estimation
    xhat(:,k+1) = A*xhat(:,k)+Bd*dhat(:,k)+B*u_rec(:,k)+Lx*(y(:,k)-C*xhat(:,k)-Cd*dhat(:,k));
    dhat(:,k+1) = dhat(:,k)+L1*(y(:,k)-C*xhat(:,k)-Cd*dhat(:,k));
end   
        

% plot values
figure(1)
plot(T_1,x(1,:),'b','LineWidth', 1.3);
hold on
plot(T,y_ref(1,:),'r','LineWidth', 1.3);
plot(T,d(1,:),'g','LineWidth', 1.3);
hold off

legend('x_b', 'x_bref', 'disturbance');

figure(2)
plot(T_1,x(3,:),'b','LineWidth', 1.3);
hold on
plot(T,y_ref(2,:),'r','LineWidth', 1.3);
plot(T,d(2,:),'g','LineWidth', 1.3);
hold off

figure(3)
hold on
for i = 5:8
    plot(T_1,x(i,:),'LineWidth', 1.3);
end
hold off

legend('a', 'da', 'g', 'dg');

%% Functions
function [T,S] = predict_model(A,B,nx,nu,N)
    % T matrix from initial state
    T = zeros(nx*(N+1),nx);
    for k = 0:N
        T(k*nx+1:(k+1)*nx,:) = A^k;
    end

    % S matrix from input
    S = zeros(nx*(N+1),nu*N);
    for k = 1:N
        for i = 0:k-1
            S(k*nx+1:(k+1)*nx,i*nu+1:(i+1)*nu) = A^(k-1-i)*B;
        end
    end
end

function [H,h] = cost_model(S,T,nx,nu,N,Q,P,R)
    Qbar = blkdiag(kron(eye(N),Q),P);
    Rbar = kron(eye(N),R);
    H = S'*Qbar*S+Rbar;   
    hx0 = S'*Qbar*T;
    hxref = -S'*Qbar*kron(ones(N+1,1),eye(nx));
    huref = -Rbar*kron(ones(N,1),eye(nu));
    h = [hx0 hxref huref];
end